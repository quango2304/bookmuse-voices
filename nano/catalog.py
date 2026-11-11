"""The voices the app lists, by app key → the file voice.py wrote, and the settings it renders with. Display names are
the owner's (2026-11-03), not the source voices'; voice.py's files keep the source names. Tuấn Kiệt (thai_son) and Thảo Vy
(kim_thanh) were removed from the apps the same day."""

# App key → the voice file voice.py wrote.
# 2026-11-04 (pauses): Nano copies the pauses of its reference clip, so clips with extra mid-phrase pauses made voices stop
# where nothing is written. Picked per voice from three kinds of clone, scored by pauses on unpunctuated text (want none) and
# at 3 marks of punctuation (want 3): -flow2 / -flow3 (a Vbee clip read without / with little punctuation, vbee_ref.py), -pc
# and -pc2 (the clip with every quiet gap cut to 120 / 220 ms: a pc.py of the session, see work/nano/pc), none (the original).
# 2026-11-04 (misreads): every clone scored by eval_cer.py (Whisper over a 12-paragraph corpus at speed 1, 4 then 8 seeds)
# and by pauses at punctuation; the one with the fewest misreads kept unless it adds 0.2 or more pauses of 300 ms a chunk.
# Changed: Ngọc Hân, Thu Hà (-flow4: a new Vbee passage, work/nano/vbee/texts/t4.txt), Bảo Trân, Hải Đăng (-flow2).
# 2026-11-05: Diệu Linh -flow3, scored on Android's 8-step int8 graphs too (18.8 % against -flow4's 27.1 %, and fewer on Core ML).
VOICES = {
    "tam_an": "thien_tam_duc-pc",
    "dang_khoa": "quang_son-pc",
    "hoang_long": "thien_minh",
    "van_thanh": "s_hochiminh_male_phapvan_book_vc-pc",
    "ba_phuc": "s_tiengiang_male_giongnamratnambo_story_vc-flow2",
    "ngoc_han": "s_hochiminh_female_machvi_story_vc-flow4",
    "trung_hieu": "s_vinhlong_male_hanhquang20260316220828233_story_vc-flow2",
    "bao_tran": "sg_female_thanhmai_yt_24k-pre-flow2",
    "thu_ha": "n_hanoi_female_nguyetnga2_book_vc-flow4",
    "dieu_linh": "n_hanoi_female_thanhphuongmaster_book_vc-flow3",
    "mai_chi": "n_hanoi_female_mchiensachnoi_book_vc-pc",
    "hai_dang": "n_namdinh_male_haichuyen20251209185109485_book_vc-flow2",
}
SETTINGS = {"steps": 16, "cfg": 3.0, "sway": -1.0}
# Each voice's speed factor to one pace (speeds.py); the new clones scaled from the old ones' on the eval corpus, so the
# pace stays. The app applies it on the audio within 0.9–1.1.
SPEEDS = {"tam_an": 1.044, "dang_khoa": 1.035, "hoang_long": 0.75, "van_thanh": 1.0, "ba_phuc": 1.168, "ngoc_han": 1.116, "trung_hieu": 1.12, "bao_tran": 1.09, "thu_ha": 0.924, "dieu_linh": 0.973, "mai_chi": 0.967, "hai_dang": 1.064}
# Android renders 8 steps where iOS renders 16, and a clone that reads well at 16 can skip words at 8: a voice can take its
# own clone there, app key → (the file voice.py wrote, its speed factor). The key stays the same, so a choice syncs.
ANDROID = {"trung_hieu": ("s_vinhlong_male_hanhquang20260316220828233_story_vc-flow3", 1.13)}
