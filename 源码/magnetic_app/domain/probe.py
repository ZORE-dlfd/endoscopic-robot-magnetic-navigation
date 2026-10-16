def risk_description(source, index):
    return (f'中心线 #{index}  ·  弧长 {source["arc_s_mm"][index]:.1f} mm\n'
            f'归一化风险 {source["risk_norm"][index]:.3f}  ·  净空 {source["clearance_mm"][index]:.2f} mm\n'
            f'曲率 {source["curvature_inv_mm"][index]:.5f} mm⁻¹\n'
            f'目标速度 {source["speed_target_mm_s"][index]:.2f} mm/s\n'
            f'风险 / 净空 / 曲率限速 {source["speed_risk_limit_mm_s"][index]:.2f} / '
            f'{source["speed_clearance_limit_mm_s"][index]:.2f} / '
            f'{source["speed_curvature_limit_mm_s"][index]:.2f} mm/s')
