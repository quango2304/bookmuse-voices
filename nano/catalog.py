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
    "minh_khang": "n_hanoi_male_giongnamtramhay_story_vc",
    "trung_hieu": "s_vinhlong_male_hanhquang20260316220828233_story_vc",
    "bao_tran": "sg_female_thanhmai_yt_24k-pre"
}
SETTINGS = {"steps": 16, "cfg": 3.0, "sway": -1.0}
# Each voice's speed factor to one pace (speeds.py).
SPEEDS = {"tam_an": 1.044, "dang_khoa": 1.035, "hoang_long": 0.75, "van_thanh": 1.039, "ba_phuc": 1.168, "ngoc_han": 1.249, "minh_khang": 0.873, "trung_hieu": 1.167, "bao_tran": 1.084}
