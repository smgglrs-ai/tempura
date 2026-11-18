fn main() {
    // replicate check
    for b in 0u16..=255 {
        let v = tempura_kernels::e4m3::e4m3_bits_to_f32(b as u8);
        if v.is_finite() {
            let back = tempura_kernels::e4m3::f32_to_e4m3_bits(v);
            if back != b as u8 { println!("b={b:#04x} v={v} back={back:#04x}"); }
        }
    }
}
