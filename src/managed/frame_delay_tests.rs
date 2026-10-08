use super::*;

fn settings(threads: u32, max_frame_delay: u32) -> Settings {
    Settings {
        threads,
        max_frame_delay,
        ..Settings::default()
    }
}

#[test]
fn auto_frame_delay_is_two_with_workers_and_one_without() {
    assert_eq!(settings(1, 0).effective_frame_delay(), 1);
    for t in [0, 2, 4, 8, 16] {
        assert_eq!(settings(t, 0).effective_frame_delay(), 2, "threads={t}");
    }
    // Explicit values pass through untouched.
    for d in [1, 2, 3, 8] {
        assert_eq!(settings(8, d).effective_frame_delay(), d as i32);
        assert_eq!(settings(1, d).effective_frame_delay(), d as i32);
    }
}

#[test]
fn decoder_context_count_follows_the_auto_rule() {
    let fc = |t, d| Decoder::with_settings(settings(t, d)).unwrap().ctx.fc.len();
    assert_eq!(fc(1, 0), 1);
    assert_eq!(fc(4, 0), 2);
    assert_eq!(fc(8, 0), 2);
    assert_eq!(fc(4, 1), 1);
    assert_eq!(fc(4, 3), 3);
    // Never more contexts than workers.
    assert_eq!(fc(2, 8), 2);
}
