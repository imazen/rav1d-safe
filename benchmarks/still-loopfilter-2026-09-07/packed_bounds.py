"""Check linear sums and the unpack permutation in the private packed kernel.

This models named intrinsic operations; it does not verify compiler lowering
or the complete filter. Narrow-filter/mask bounds remain the written argument,
and actual machine output is independently compared with the scalar decoder.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--wide-kernel', '--wide-prototype', dest='wide_kernel', type=Path)
a = p.parse_args()
path = a.repo / 'src/safe_simd/loopfilter_packed6.rs'
source = path.read_text()
record = dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              scope='Linear sums and unpack permutation model, not a compiler or full decoder proof')


def sums(text, taps, start, end, outputs, maximum):
    class Linear:
        def __init__(self, coeff):
            self.coeff = coeff

        def __add__(self, other):
            return Linear([x + y for x, y in zip(self.coeff, other.coeff)])

        def times(self, n):
            return Linear([x * n for x in self.coeff])

        def bound(self):
            lo = sum(min(0, x * 255) for x in self.coeff[:-1]) + self.coeff[-1]
            hi = sum(max(0, x * 255) for x in self.coeff[:-1]) + self.coeff[-1]
            return [lo, hi]

    count = len(taps)
    ctx = {name: Linear([int(i == j) for j in range(count)] + [0])
           for i, name in enumerate(taps)}
    ctx.update({f'c{n}': Linear([0] * count + [n]) for n in [4, 8]})
    bounds, found = [], {}

    def calc(node):
        if isinstance(node, ast.Name):
            return ctx[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        assert isinstance(node, ast.Call), ast.dump(node)
        args = [calc(arg) for arg in node.args]
        name = node.func.id
        if name in ['add', '_mm_add_epi16', 'add3', 'add4']:
            value = args[0]
            for arg in args[1:]:
                value = value + arg
        elif name in ['dbl', 'triple', '_mm_slli_epi16_1']:
            value = args[0].times(3 if name == 'triple' else 2)
        elif name == '_mm_set1_epi16':
            value = Linear([0] * count + args)
        elif name in ['_mm_srai_epi16_3', '_mm_srai_epi16_4']:
            return args[0], int(name[-1])
        else:
            raise AssertionError(name)
        bounds.append(value.bound())
        return value

    block = text[text.index(start):text.index(end)]
    block = re.sub(r'//[^\n]*', '', block)
    block = re.sub(r'let (dbl|triple) = \|.*?;', '', block, flags=re.S)
    for stmt in block.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue
        name, expr = stmt.split('=', 1)
        name = re.sub(r'^let\s+(mut\s+)?', '', name.strip())
        expr = re.sub(r'::<(\d+)>', r'_\1', expr.strip())
        value = calc(ast.parse(expr, mode='eval').body)
        if isinstance(value, tuple):
            linear, shift = value
            coeff = linear.coeff
            assert all(c >= 0 for c in coeff), (name, coeff)
            assert sum(coeff[:-1]) == 1 << shift, (name, coeff)
            assert coeff[-1] == 1 << (shift - 1), (name, coeff)
            found[name] = dict(coefficients=dict(zip(taps + ['constant'], coeff)),
                               shift=shift, pre_shift_bound=linear.bound())
        else:
            ctx[name] = value
    assert len(found) == outputs, found
    assert min(b[0] for b in bounds) >= 0
    assert max(b[1] for b in bounds) == maximum
    return dict(outputs=found, weighted_intermediate_range=[0, maximum])


record['six_tap'] = sums(source, ['p2_v', 'p1_v', 'p0_v', 'q0_v', 'q1_v', 'q2_v'],
                         'let p2_3 =', 'let neg128 =', 4, 2044)

# Unpack operates on groups of 1, 2, or 4 original i16 lanes. Tracking every
# distinct input lane through the actual source's network proves the modeled
# permutation for arbitrary values, rather than one random matrix.
def verify_transpose(text):
    section = text[text.index('fn transpose8'):]
    transpose = section[:section.index('\n    }')]
    ctx = {'m': [[(r, c) for c in range(8)] for r in range(8)]}


    def permute(node):
        if isinstance(node, ast.Name):
            return ctx[node.id]
        if isinstance(node, ast.Subscript):
            return ctx[node.value.id][node.slice.value]
        assert isinstance(node, ast.Call), ast.dump(node)
        match = re.fullmatch(r'_mm_unpack(lo|hi)_epi(16|32|64)', node.func.id)
        assert match, ast.dump(node)
        group = int(match[2]) // 16
        x, y = [permute(arg) for arg in node.args]
        begin = 0 if match[1] == 'lo' else 4
        out = []
        for i in range(begin, begin + 4, group):
            out += x[i:i + group] + y[i:i + group]
        return out


    for name, expr in re.findall(r'let (\w+) = ([^;]+);', transpose):
        ctx[name] = permute(ast.parse(expr, mode='eval').body)
    tail = transpose.rsplit(';', 1)[1]
    exprs = re.findall(r'_mm_unpack(?:lo|hi)_epi64\([^)]*\)', tail)
    actual = [permute(ast.parse(expr, mode='eval').body) for expr in exprs]
    expected = [[(r, c) for r in range(8)] for c in range(8)]
    assert actual == expected
    return dict(verified_input_lanes=64, permutation=actual)


record['transpose'] = verify_transpose(source)

if a.wide_kernel:
    wide = a.wide_kernel.read_text()
    taps = [f'{side}{i}_v' for side, indices in [('p', range(6, -1, -1)), ('q', range(7))]
            for i in indices]
    record['wide_kernel'] = dict(
        source=str(a.wide_kernel), sha256=hashlib.sha256(a.wide_kernel.read_bytes()).hexdigest(),
        analysis=sums(wide, taps, 'let p6_5 =', '// Narrow filter', 18, 4088),
        transpose=verify_transpose(wide))

with a.output.open('x') as output:
    output.write(json.dumps(record, indent=2) + '\n')
print('Verified six-tap weighted bounds and all 64 transpose lanes.')
if a.wide_kernel:
    print('Verified 12 wide and 6 mid-filter weight sums and the wide-kernel transpose.')
