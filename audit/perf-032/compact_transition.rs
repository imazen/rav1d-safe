use rav1d_disjoint_mut::{DisjointMut, set_parallelism};
#[test]
fn grows_after_parallelism_change_and_preserves_exclusion() {
    let mut buf=DisjointMut::new_eager(vec![0u8;32]);
    buf.index_mut(0..2).copy_from_slice(&[17,23]);
    set_parallelism(8);
    buf.resize(131072,0);
    { let mut tail=buf.index_mut(131070..131072);tail.copy_from_slice(&[31,47]); }
    let guard=buf.index_mut(65530..65540);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(||buf.index_mut(65535..65545))).is_err());
    drop(guard);
    assert_eq!(&*buf.index(0..2),&[17,23]);
    assert_eq!(&*buf.index(131070..131072),&[31,47]);
    buf.resize(16,0);
    assert_eq!(&*buf.index(0..2),&[17,23]);
    buf.resize(262144,0);
    *buf.index_mut(262143)=59;
    assert_eq!(*buf.index(262143),59);
}
