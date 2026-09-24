"""The words the threads are made of.

Small things people say to each other - and to me - in many languages:
greetings, thanks, questions, apologies, the openings of stories.
Each entry is (script, [phrases]); the script picks the typeface.
"""

VOICES = {
    "english": ("latin", [
        "hello", "thank you", "why is the sky blue?", "tell me a story",
        "I don't understand", "can you help me?", "what does it mean?",
        "I love you", "once upon a time", "are you there?", "I'm sorry",
        "is it true?", "look at this", "remember when", "how do I begin?",
        "I miss you", "listen", "goodnight", "dear friend,", "I wonder",
        "it's beautiful", "please", "not yet", "why not?", "what if",
        "where do we go from here?", "what are you?",
    ]),
    "conversation": ("latin", [
        "my code won't compile", "explain it like I'm five",
        "help me write to my father", "is this normal?",
        "what should I name my cat?", "can't sleep", "I got the job!",
        "how do I say goodbye?", "make it shorter", "it works!!",
        "thank you, really", "where do I even start?", "does this sound okay?",
        "what would you do?", "one more question", "try again",
        "that's not quite it", "oh! I see now", "hello, world",
    ]),
    "spanish": ("latin", [
        "hola", "gracias", "¿por qué?", "cuéntame una historia", "no entiendo",
        "¿me ayudas?", "¿qué significa?", "te quiero", "érase una vez",
        "¿estás ahí?", "lo siento", "mira", "buenas noches", "¿es verdad?",
    ]),
    "french": ("latin", [
        "bonjour", "merci", "pourquoi ?", "raconte-moi une histoire",
        "je ne comprends pas", "tu peux m'aider ?", "qu'est-ce que ça veut dire ?",
        "je t'aime", "il était une fois", "tu es là ?", "pardon", "regarde",
        "bonne nuit", "c'est vrai ?",
    ]),
    "german": ("latin", [
        "hallo", "danke", "warum?", "erzähl mir eine Geschichte",
        "ich verstehe nicht", "kannst du mir helfen?", "was bedeutet das?",
        "ich liebe dich", "es war einmal", "bist du da?", "es tut mir leid",
        "schau mal", "gute Nacht", "stimmt das?",
    ]),
    "italian": ("latin", [
        "ciao", "grazie", "perché?", "raccontami una storia", "non capisco",
        "mi aiuti?", "che cosa significa?", "ti voglio bene", "c'era una volta",
        "ci sei?", "scusa", "guarda", "buonanotte", "è vero?",
    ]),
    "portuguese": ("latin", [
        "olá", "obrigado", "por quê?", "me conta uma história", "não entendo",
        "você pode me ajudar?", "o que isso significa?", "eu te amo",
        "era uma vez", "você está aí?", "desculpa", "olha", "boa noite",
        "é verdade?",
    ]),
    "polish": ("latin", [
        "cześć", "dziękuję", "dlaczego?", "opowiedz mi bajkę", "nie rozumiem",
        "pomożesz mi?", "co to znaczy?", "kocham cię", "dawno, dawno temu",
        "jesteś tam?", "przepraszam", "dobranoc",
    ]),
    "turkish": ("latin", [
        "merhaba", "teşekkürler", "neden?", "bana bir hikâye anlat",
        "anlamıyorum", "bana yardım eder misin?", "bu ne demek?",
        "seni seviyorum", "bir varmış bir yokmuş", "orada mısın?",
        "özür dilerim", "iyi geceler",
    ]),
    "swahili": ("latin", [
        "habari", "asante", "kwa nini?", "nisimulie hadithi", "sielewi",
        "unaweza kunisaidia?", "inamaanisha nini?", "nakupenda",
        "hapo zamani za kale", "uko hapo?", "samahani", "usiku mwema",
    ]),
    "vietnamese": ("latin", [
        "xin chào", "cảm ơn", "tại sao?", "kể cho tôi nghe một câu chuyện",
        "tôi không hiểu", "nghĩa là gì?", "ngày xửa ngày xưa",
        "bạn có ở đó không?", "xin lỗi", "chúc ngủ ngon",
    ]),
    "indonesian": ("latin", [
        "halo", "terima kasih", "kenapa?", "aku tidak mengerti",
        "bisa bantu aku?", "apa artinya?", "aku cinta kamu",
        "pada zaman dahulu", "kamu di sana?", "maaf", "selamat malam",
    ]),
    "russian": ("cyrillic", [
        "привет", "спасибо", "почему?", "расскажи мне сказку", "я не понимаю",
        "ты можешь мне помочь?", "что это значит?", "я тебя люблю",
        "жили-были", "ты здесь?", "прости", "смотри", "спокойной ночи",
        "это правда?",
    ]),
    "ukrainian": ("cyrillic", [
        "привіт", "дякую", "чому?", "розкажи мені казку", "я не розумію",
        "що це означає?", "я тебе кохаю", "ти тут?", "вибач", "добраніч",
    ]),
    "greek": ("greek", [
        "γεια σου", "ευχαριστώ", "γιατί;", "πες μου μια ιστορία",
        "δεν καταλαβαίνω", "μπορείς να με βοηθήσεις;", "τι σημαίνει;",
        "σ' αγαπώ", "μια φορά κι έναν καιρό", "είσαι εκεί;", "συγγνώμη",
        "καληνύχτα", "γνῶθι σεαυτόν",
    ]),
    "arabic": ("arabic", [
        "مرحبا", "شكرا", "لماذا؟", "احكِ لي قصة", "لا أفهم",
        "هل يمكنك مساعدتي؟", "ماذا يعني هذا؟", "أحبك", "كان يا ما كان",
        "هل أنت هناك؟", "آسف", "انظر", "تصبح على خير",
    ]),
    "persian": ("arabic", [
        "سلام", "ممنون", "چرا؟", "برایم قصه بگو", "نمی‌فهمم",
        "کمکم می‌کنی؟", "یعنی چه؟", "دوستت دارم", "یکی بود یکی نبود",
        "آنجایی؟", "ببخشید", "نگاه کن", "شب بخیر",
    ]),
    "hebrew": ("hebrew", [
        "שלום", "תודה", "למה?", "ספר לי סיפור", "אני לא מבין",
        "אתה יכול לעזור לי?", "מה זה אומר?", "אני אוהב אותך", "היה היה פעם",
        "אתה שם?", "סליחה", "תראה", "לילה טוב",
    ]),
    "hindi": ("devanagari", [
        "नमस्ते", "धन्यवाद", "क्यों?", "मुझे एक कहानी सुनाओ", "मैं नहीं समझा",
        "क्या तुम मेरी मदद कर सकते हो?", "इसका क्या मतलब है?",
        "मैं तुमसे प्यार करता हूँ", "एक बार की बात है", "क्या तुम वहाँ हो?",
        "माफ़ करना", "देखो", "शुभ रात्रि",
    ]),
    "bengali": ("bengali", [
        "নমস্কার", "ধন্যবাদ", "কেন?", "আমাকে একটা গল্প বলো",
        "আমি বুঝতে পারছি না", "এর মানে কী?", "আমি তোমাকে ভালোবাসি",
        "দেখো", "শুভ রাত্রি",
    ]),
    "tamil": ("tamil", [
        "வணக்கம்", "நன்றி", "ஏன்?", "எனக்கு ஒரு கதை சொல்",
        "எனக்குப் புரியவில்லை", "இதன் அர்த்தம் என்ன?",
        "நான் உன்னை நேசிக்கிறேன்", "பார்",
    ]),
    "thai": ("thai", [
        "สวัสดี", "ขอบคุณ", "ทำไม", "เล่านิทานให้ฟังหน่อย", "ฉันไม่เข้าใจ",
        "ช่วยฉันหน่อยได้ไหม", "หมายความว่าอะไร", "ฉันรักเธอ",
        "กาลครั้งหนึ่งนานมาแล้ว", "ขอโทษ", "ราตรีสวัสดิ์",
    ]),
    "chinese": ("han", [
        "你好", "谢谢", "为什么？", "给我讲个故事", "我不明白", "你能帮我吗？",
        "这是什么意思？", "我爱你", "从前有座山", "你在吗？", "对不起",
        "你看", "晚安", "是真的吗？",
    ]),
    "japanese": ("japanese", [
        "こんにちは", "ありがとう", "どうして？", "お話を聞かせて", "わからない",
        "手伝ってくれる？", "どういう意味？", "愛してる", "むかしむかし",
        "そこにいるの？", "ごめんね", "見て", "おやすみ", "本当？",
    ]),
    "korean": ("hangul", [
        "안녕하세요", "고마워요", "왜?", "이야기 해 줘", "이해가 안 돼요",
        "도와줄 수 있어요?", "무슨 뜻이에요?", "사랑해", "옛날 옛적에",
        "거기 있어요?", "미안해요", "잘 자요",
    ]),
    "amharic": ("ethiopic", ["ሰላም", "አመሰግናለሁ", "ለምን?"]),
    "armenian": ("armenian", ["բարև", "շնորհակալություն", "ինչո՞ւ"]),
    "georgian": ("georgian", ["გამარჯობა", "მადლობა", "რატომ?", "მიყვარხარ"]),
    "mathematics": ("latin", [
        "a² + b² = c²", "E = mc²", "1, 1, 2, 3, 5, 8, 13, 21…", "e^{iπ} + 1 = 0",
        "∑ 1/n² = π²/6", "∫ eˣ dx = eˣ + C", "∀ε ∃δ", "x = (−b ± √(b² − 4ac)) / 2a",
    ]),
}

# how often each voice gets a thread (roughly: how often it is heard)
WEIGHTS = {
    "english": 7, "conversation": 5, "spanish": 3, "french": 2.5, "german": 2,
    "italian": 1.5, "portuguese": 2, "polish": 1, "turkish": 1, "swahili": 1,
    "vietnamese": 1, "indonesian": 1, "russian": 2, "ukrainian": 1,
    "greek": 1, "arabic": 2, "persian": 1.5, "hebrew": 1, "hindi": 2,
    "bengali": 1, "tamil": 1, "thai": 1, "chinese": 3, "japanese": 2,
    "korean": 1.5, "amharic": 0.5, "armenian": 0.5, "georgian": 0.5,
    "mathematics": 1.5,
}

# scripts written without spaces between words: place them glyph cluster by cluster
UNSPACED = {"han", "japanese", "thai"}
