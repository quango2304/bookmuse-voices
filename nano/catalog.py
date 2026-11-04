"""The voices the app lists, by app key → the file voice.py wrote, and the settings it renders with. Display names are
the owner's (2026-11-03), not the source voices'; voice.py's files keep the source names. Tuấn Kiệt (thai_son) and Thảo Vy
(kim_thanh) were removed from the apps the same day."""

# App key → the voice file voice.py wrote.
VOICES = {
    "tam_an": "thien_tam_duc",
    "dang_khoa": "quang_son",
    "hoang_long": "thien_minh",
    "van_thanh": "s_hochiminh_male_phapvan_book_vc",
    "ba_phuc": "s_tiengiang_male_giongnamratnambo_story_vc",
    "ngoc_han": "s_hochiminh_female_machvi_story_vc",
    "minh_khang": "n_hanoi_male_giongnamtramhay_story_vc-flow2",
    "trung_hieu": "s_vinhlong_male_hanhquang20260316220828233_story_vc",
    "bao_tran": "sg_female_thanhmai_yt_24k-pre",
    # 2026-11-04: cloned from Vbee clips read without punctuation (-flow2: no pauses to learn).
    "thu_ha": "n_hanoi_female_nguyetnga2_book_vc-flow2",
    "dieu_linh": "n_hanoi_female_thanhphuongmaster_book_vc-flow2",
    "mai_chi": "n_hanoi_female_mchiensachnoi_book_vc-flow2",
    "hai_dang": "n_namdinh_male_haichuyen20251209185109485_book_vc-flow2",
}
SETTINGS = {"steps": 16, "cfg": 3.0, "sway": -1.0}
# Each voice's speed factor to one pace (speeds.py).
SPEEDS = {"tam_an": 1.044, "dang_khoa": 1.035, "hoang_long": 0.75, "van_thanh": 1.039, "ba_phuc": 1.168, "ngoc_han": 1.249, "minh_khang": 0.857, "trung_hieu": 1.167, "bao_tran": 1.084, "thu_ha": 0.912, "dieu_linh": 0.988, "mai_chi": 0.967, "hai_dang": 0.972}
