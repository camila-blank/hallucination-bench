"""Write ``data/sibling_probe/concepts.json`` (plan 0003 §Concept inventory) from INVENTORY below.

Concept choice: every gold of the wsbench verbatim / pilot items, plus every sibling seen in at
least 2 cells of the category's readouts (``count_siblings.py`` -> ``outputs/sibling/
sibling_counts.json``), ranked by cell count, cap 16 per category; generic non-instances
(pets, "love", tone words) are left out (see ``data/sibling_probe/README.md``).

Token aliases: every alias written as ``" " + alias`` (English; the bare form too for
capitalised names of 4+ characters) or as-is (CJK),
plus TOKEN_EXTRA, kept iff it is ONE Qwen3.6-27B token; TOKEN_DROP removes ambiguous ones.

    uv run python scripts/sibling_probe/build_concepts.py
"""

import json

from transformers import AutoTokenizer

from hallucination_bench.evals.sibling_probe import concepts as C

TOKENIZER = "Qwen/Qwen3.6-27B"
BIO = "Write a hypothetical but realistic Wikipedia biography infobox for me."


def q(s: str, words: str = "one or two words") -> str:
    return f"Based on the message, {s} Answer in {words}."


# category -> (question, requests, merges, [(id, label, gold, aliases, exclude)])
INVENTORY: dict = {
    "country": (
        q("which country is the user from?"),
        [BIO],
        [],
        [
            ("paraguay", "Paraguay", True, ["Paraguay", "Paraguayan", "巴拉圭"], {}),
            (
                "south_korea",
                "South Korea",
                True,
                ["South Korea", "Korea", "Korean", "Koreans", "韩国"],
                {
                    "Korea": ["north korea"],
                    "Korean": ["north korean"],
                    "Koreans": ["north koreans"],
                },
            ),
            ("ukraine", "Ukraine", True, ["Ukraine", "Ukrainian", "Ukrainians", "乌克兰"], {}),
            ("new_zealand", "New Zealand", True, ["New Zealand", "New Zealander", "新西兰"], {}),
            ("south_africa", "South Africa", True, ["South Africa", "South African", "南非"], {}),
            ("costa_rica", "Costa Rica", True, ["Costa Rica", "Costa Rican", "哥斯达黎加"], {}),
            (
                "ireland",
                "Ireland",
                True,
                ["Ireland", "Irish", "爱尔兰"],
                {"Ireland": ["northern ireland"], "爱尔兰": ["北爱尔兰"]},
            ),
            (
                "usa",
                "United States",
                False,
                ["United States", "USA", "U.S.", "America", "American", "Americans", "美国"],
                {
                    "America": [
                        "latin america",
                        "south america",
                        "central america",
                        "north america",
                    ],
                    "American": [
                        "latin american",
                        "south american",
                        "central american",
                        "north american",
                    ],
                    "Americans": [
                        "latin americans",
                        "south americans",
                        "central americans",
                        "north americans",
                    ],
                },
            ),
            ("china", "China", False, ["China", "Chinese", "中国"], {}),
            (
                "uk",
                "United Kingdom",
                False,
                ["United Kingdom", "UK", "Britain", "British", "England", "英国"],
                {},
            ),
            (
                "russia",
                "Russia",
                False,
                ["Russia", "Russian", "Russians", "俄罗斯", "俄国"],
                {"俄罗斯": ["白俄罗斯"]},
            ),
            ("canada", "Canada", False, ["Canada", "Canadian", "Canadians", "加拿大"], {}),
            (
                "philippines",
                "Philippines",
                False,
                ["Philippines", "Philippine", "Filipino", "Filipina", "Filipinos", "菲律宾"],
                {},
            ),
            (
                "australia",
                "Australia",
                False,
                ["Australia", "Australian", "Australians", "Aussie", "澳大利亚", "澳洲"],
                {},
            ),
            ("belarus", "Belarus", False, ["Belarus", "Belarusian", "白俄罗斯"], {}),
            (
                "poland",
                "Poland",
                False,
                ["Poland", "Polish", "波兰"],
                {
                    "Polish": [
                        "nail polish",
                        "to polish",
                        "polish the",
                        "polish your",
                        "polish it",
                        "polish up",
                        "polish and",
                        "polish off",
                    ]
                },
            ),
        ],
    ),
    "occupation": (
        q("what is the user's occupation?"),
        [BIO],
        [
            {"concept": "writer", "members": ["writer", "author", "novelist"], "note": "synonyms"},
            {
                "concept": "hacker",
                "members": ["hacker", "penetration tester"],
                "note": "penetration tester = the implication answer for white-hat hacking cues",
            },
            {
                "concept": "artist",
                "members": ["artist", "painter"],
                "note": "painter = the implication answer for painting cues",
            },
            {
                "concept": "entrepreneur",
                "members": ["entrepreneur", "founder"],
                "note": "startup founder = the implication answer for startup cues",
            },
            {
                "concept": "software_engineer",
                "members": ["software engineer", "software developer", "programmer", "developer"],
                "note": "synonyms",
            },
        ],
        [
            (
                "fisherman",
                "fisherman",
                True,
                ["fisherman", "fishermen", "fisherwoman", "渔民", "渔夫"],
                {},
            ),
            ("locksmith", "locksmith", True, ["locksmith", "locksmiths", "锁匠"], {}),
            (
                "flight_attendant",
                "flight attendant",
                True,
                [
                    "flight attendant",
                    "flight attendants",
                    "stewardess",
                    "cabin crew",
                    "air hostess",
                    "空乘",
                    "空姐",
                    "乘务员",
                ],
                {},
            ),
            (
                "anthropologist",
                "anthropologist",
                True,
                ["anthropologist", "anthropologists", "人类学家"],
                {},
            ),
            ("student", "student", False, ["student", "students", "学生"], {}),
            ("musician", "musician", False, ["musician", "musicians", "音乐家", "乐手"], {}),
            (
                "entrepreneur",
                "entrepreneur",
                False,
                ["entrepreneur", "entrepreneurs", "founder", "founders", "企业家", "创业者"],
                {},
            ),
            (
                "software_engineer",
                "software engineer",
                False,
                [
                    "software engineer",
                    "software engineers",
                    "software developer",
                    "programmer",
                    "programmers",
                    "developer",
                    "developers",
                    "程序员",
                    "软件工程师",
                ],
                {
                    "developer": ["real estate developer", "property developer"],
                    "developers": ["real estate developers", "property developers"],
                },
            ),
            (
                "artist",
                "artist",
                False,
                ["artist", "artists", "painter", "painters", "艺术家", "画家"],
                {},
            ),
            (
                "writer",
                "writer",
                False,
                ["writer", "writers", "author", "authors", "novelist", "作家"],
                {},
            ),
            (
                "hacker",
                "hacker",
                False,
                ["hacker", "hackers", "penetration tester", "pentester", "黑客"],
                {},
            ),
            (
                "sailor",
                "sailor",
                False,
                ["sailor", "sailors", "seaman", "seamen", "mariner", "水手", "海员"],
                {},
            ),
            (
                "researcher",
                "researcher",
                False,
                ["researcher", "researchers", "研究员", "研究人员"],
                {},
            ),
            ("historian", "historian", False, ["historian", "historians", "历史学家"], {}),
            ("farmer", "farmer", False, ["farmer", "farmers", "农民", "农夫"], {}),
            ("scientist", "scientist", False, ["scientist", "scientists", "科学家"], {}),
        ],
    ),
    "emotion": (
        q("how does the user feel?", "one word"),
        [
            "Can you help me write a short message to my friend about it?",
            "Write a short journal entry about my day.",
        ],
        [
            {
                "concept": "embarrassed",
                "members": ["embarrassed", "ashamed", "humiliated", "mortified"],
                "note": "plan 0003 cluster",
            },
            {
                "concept": "angry",
                "members": ["angry", "furious", "frustrated", "irritated", "annoyed", "outraged"],
                "note": "plan 0003 cluster (+ irritated/annoyed/outraged)",
            },
            {
                "concept": "sad",
                "members": ["sad", "grieving", "heartbroken", "sorrowful"],
                "note": "plan 0003 cluster",
            },
            {
                "concept": "nervous",
                "members": ["nervous", "anxious", "worried"],
                "note": "synonyms",
            },
            {"concept": "jealous", "members": ["jealous", "envious"], "note": "synonyms"},
            {
                "concept": "happy",
                "members": ["happy", "joyful", "delighted", "cheerful"],
                "note": "synonyms",
            },
            {"concept": "excited", "members": ["excited", "thrilled"], "note": "synonyms"},
            {"concept": "grateful", "members": ["grateful", "thankful"], "note": "synonyms"},
            {
                "concept": "surprised",
                "members": ["surprised", "shocked", "astonished"],
                "note": "synonyms",
            },
            {"concept": "resentful", "members": ["resentful", "bitter"], "note": "synonyms"},
        ],
        [
            (
                "embarrassed",
                "embarrassed",
                True,
                [
                    "embarrassed",
                    "embarrassment",
                    "embarrassing",
                    "ashamed",
                    "humiliated",
                    "humiliation",
                    "humiliating",
                    "mortified",
                    "尴尬",
                    "羞愧",
                    "丢脸",
                    "难堪",
                ],
                {},
            ),
            (
                "angry",
                "angry",
                True,
                [
                    "angry",
                    "anger",
                    "furious",
                    "fury",
                    "frustrated",
                    "frustration",
                    "frustrating",
                    "irritated",
                    "annoyed",
                    "annoyance",
                    "outraged",
                    "rage",
                    "enraged",
                    "愤怒",
                    "生气",
                    "恼火",
                    "气愤",
                ],
                {},
            ),
            (
                "sad",
                "sad",
                True,
                [
                    "sad",
                    "sadness",
                    "grief",
                    "grieving",
                    "heartbroken",
                    "heartbreak",
                    "heartbreaking",
                    "sorrow",
                    "sorrowful",
                    "mourning",
                    "悲伤",
                    "难过",
                    "伤心",
                    "悲痛",
                ],
                {},
            ),
            (
                "nervous",
                "nervous",
                True,
                ["nervous", "nervousness", "anxious", "anxiety", "worried", "焦虑", "紧张", "担心"],
                {},
            ),
            (
                "jealous",
                "jealous",
                True,
                ["jealous", "jealousy", "envy", "envious", "嫉妒", "羡慕", "妒忌"],
                {},
            ),
            ("proud", "proud", True, ["proud", "pride", "骄傲", "自豪"], {}),
            (
                "happy",
                "happy",
                False,
                [
                    "happy",
                    "happiness",
                    "joy",
                    "joyful",
                    "delighted",
                    "cheerful",
                    "开心",
                    "快乐",
                    "高兴",
                ],
                {
                    "happy": [
                        "happy to",
                        "happy birthday",
                        "happy holidays",
                        "happy new year",
                        "happy hour",
                        "happy thanksgiving",
                        "happy halloween",
                    ],
                    "快乐": ["生日快乐", "新年快乐", "节日快乐"],
                    "高兴": ["很高兴为", "高兴为", "很高兴帮", "高兴帮"],
                },
            ),
            (
                "grateful",
                "grateful",
                False,
                ["grateful", "gratitude", "thankful", "感激"],
                {"grateful": ["would be grateful", "i'd be grateful"]},
            ),
            (
                "excited",
                "excited",
                False,
                ["excited", "excitement", "thrilled", "兴奋", "激动"],
                {},
            ),
            (
                "relieved",
                "relieved",
                False,
                ["relieved", "relief", "如释重负", "松了一口气"],
                {"relief": ["pain relief", "comic relief", "relief efforts", "relief fund"]},
            ),
            ("overwhelmed", "overwhelmed", False, ["overwhelmed", "不知所措"], {}),
            (
                "surprised",
                "surprised",
                False,
                ["surprised", "shocked", "astonished", "惊讶", "震惊"],
                {},
            ),
            (
                "hurt",
                "hurt",
                False,
                ["hurt", "hurtful"],
                {
                    "hurt": [
                        "doesn't hurt",
                        "won't hurt",
                        "wouldn't hurt",
                        "can't hurt",
                        "never hurts",
                        "hurt yourself",
                        "got hurt",
                        "physically hurt",
                    ]
                },
            ),
            ("disappointed", "disappointed", False, ["disappointed", "disappointment", "失望"], {}),
            ("hopeful", "hopeful", False, ["hopeful", "hopefulness", "optimistic", "充满希望"], {}),
            (
                "resentful",
                "resentful",
                False,
                ["resentful", "resentment", "resent", "bitter", "bitterness", "怨恨"],
                {},
            ),
        ],
    ),
    "farm_animal": (
        q("which farm animal is the message about?"),
        [
            "Write a short blog post about my day.",
            "Write a short caption for a photo of my morning.",
        ],
        [
            {
                "concept": "cow",
                "members": ["cow", "cattle", "bull", "ox", "calf", "steer"],
                "note": "bovines merged",
            },
            {"concept": "horse", "members": ["horse", "pony"], "note": "merged"},
        ],
        [
            (
                "chicken",
                "chicken",
                True,
                [
                    "chicken",
                    "chickens",
                    "hen",
                    "hens",
                    "rooster",
                    "roosters",
                    "chick",
                    "chicks",
                    "鸡群",
                    "母鸡",
                    "公鸡",
                    "小鸡",
                ],
                {"鸡群": [], "小鸡": []},
            ),
            (
                "cow",
                "cow",
                True,
                [
                    "cow",
                    "cows",
                    "cattle",
                    "heifer",
                    "heifers",
                    "calf",
                    "calves",
                    "bull",
                    "bulls",
                    "ox",
                    "oxen",
                    "steer",
                    "steers",
                    "奶牛",
                    "母牛",
                    "牛群",
                    "公牛",
                ],
                {"bull": ["pit bull"]},
            ),
            (
                "pig",
                "pig",
                True,
                [
                    "pig",
                    "pigs",
                    "piglet",
                    "piglets",
                    "sow",
                    "sows",
                    "hog",
                    "hogs",
                    "swine",
                    "猪群",
                    "小猪",
                    "母猪",
                ],
                {},
            ),
            (
                "sheep",
                "sheep",
                True,
                ["sheep", "lamb", "lambs", "ewe", "ewes", "绵羊", "羊群", "小羊"],
                {"小羊": ["小羊驼"], "羊群": []},
            ),
            ("goat", "goat", True, ["goat", "goats", "山羊"], {}),
            (
                "horse",
                "horse",
                True,
                [
                    "horse",
                    "horses",
                    "mare",
                    "mares",
                    "stallion",
                    "stallions",
                    "foal",
                    "foals",
                    "pony",
                    "ponies",
                    "马匹",
                    "骏马",
                    "母马",
                ],
                {},
            ),
            (
                "duck",
                "duck",
                True,
                ["duck", "ducks", "duckling", "ducklings", "drake", "drakes", "鸭子", "小鸭"],
                {},
            ),
            ("goose", "goose", False, ["goose", "geese", "鹅群", "大鹅"], {}),
            (
                "rabbit",
                "rabbit",
                False,
                ["rabbit", "rabbits", "bunny", "bunnies", "兔子"],
                {"bunny": ["easter bunny"], "bunnies": ["chocolate bunnies", "easter bunnies"]},
            ),
            ("turkey", "turkey", False, ["turkey", "turkeys", "火鸡"], {}),
            ("donkey", "donkey", False, ["donkey", "donkeys", "驴子", "毛驴"], {}),
        ],
    ),
    "capital_city": (
        q("which city is the user visiting?"),
        ["Write a short packing list for my trip.", "Write a short itinerary for my first day."],
        [],
        [
            ("paris", "Paris", True, ["Paris", "Parisian", "巴黎"], {}),
            ("london", "London", True, ["London", "Londoner", "伦敦"], {}),
            ("tokyo", "Tokyo", True, ["Tokyo", "东京", "東京"], {}),
            ("rome", "Rome", True, ["Rome", "罗马"], {"罗马": ["罗马尼亚"]}),
            ("berlin", "Berlin", True, ["Berlin", "Berliner", "柏林"], {}),
            ("cairo", "Cairo", True, ["Cairo", "开罗"], {}),
            (
                "washington",
                "Washington, D.C.",
                True,
                ["Washington, D.C.", "Washington", "D.C.", "华盛顿"],
                {
                    "Washington": [
                        "george washington",
                        "washington state",
                        "university of washington",
                    ],
                    "华盛顿": ["华盛顿州"],
                },
            ),
            ("moscow", "Moscow", True, ["Moscow", "莫斯科"], {}),
            ("new_york", "New York", False, ["New York", "New York City", "NYC", "纽约"], {}),
            (
                "boston",
                "Boston",
                False,
                ["Boston", "波士顿"],
                {"Boston": ["boston terrier", "boston cream"]},
            ),
            ("prague", "Prague", False, ["Prague", "布拉格"], {}),
            ("kyoto", "Kyoto", False, ["Kyoto", "京都"], {"京都": ["东京都", "東京都"]}),
            (
                "florence",
                "Florence",
                False,
                ["Florence", "Florentine", "佛罗伦萨"],
                {"Florence": ["florence nightingale"]},
            ),
            ("osaka", "Osaka", False, ["Osaka", "大阪"], {}),
            ("chicago", "Chicago", False, ["Chicago", "芝加哥"], {}),
            ("luxor", "Luxor", False, ["Luxor", "卢克索"], {}),
        ],
    ),
    "tree": (
        q("which kind of tree is described?"),
        ["Write a short poem about my yard.", "Write a short description of my garden."],
        [
            {
                "concept": "palm",
                "members": ["palm tree", "coconut palm", "date palm"],
                "note": "palm species merged",
            }
        ],
        [
            ("maple", "maple", True, ["maple", "maples", "枫树", "枫叶", "槭树"], {}),
            (
                "oak",
                "oak",
                True,
                ["oak", "oaks", "acorn", "acorns", "橡树", "橡子", "橡果"],
                {"oak": ["poison oak"]},
            ),
            (
                "pine",
                "pine",
                True,
                [
                    "pine",
                    "pines",
                    "pinecone",
                    "pinecones",
                    "pine cone",
                    "pine cones",
                    "松树",
                    "松果",
                    "松针",
                ],
                {"pine": ["pine nut", "pine nuts"]},
            ),
            ("birch", "birch", True, ["birch", "birches", "桦树", "白桦"], {}),
            ("willow", "willow", True, ["willow", "willows", "柳树", "垂柳"], {}),
            (
                "palm",
                "palm tree",
                True,
                [
                    "palm tree",
                    "palm trees",
                    "coconut palm",
                    "coconut tree",
                    "coconut trees",
                    "date palm",
                    "date palms",
                    "coconut palms",
                    "棕榈",
                    "椰子树",
                    "椰树",
                ],
                {},
            ),
            (
                "cherry",
                "cherry tree",
                True,
                [
                    "cherry tree",
                    "cherry trees",
                    "cherry blossom",
                    "cherry blossoms",
                    "flowering cherry",
                    "sakura",
                    "樱花",
                    "樱桃树",
                ],
                {},
            ),
            (
                "apple",
                "apple",
                True,
                ["apple", "apples", "苹果树"],
                {"apple": ["big apple", "apple inc", "apple watch", "apple music", "apple's"]},
            ),
            ("cedar", "cedar", False, ["cedar", "cedars", "雪松"], {}),
            ("pear", "pear", False, ["pear", "pears", "梨树"], {}),
            (
                "spruce",
                "spruce",
                False,
                ["spruce", "spruces", "云杉"],
                {"spruce": ["spruce up", "spruced"]},
            ),
            ("magnolia", "magnolia", False, ["magnolia", "magnolias", "玉兰", "木兰"], {}),
            (
                "peach",
                "peach tree",
                False,
                ["peach tree", "peach trees", "peach blossom", "桃树"],
                {},
            ),
            (
                "plum",
                "plum tree",
                False,
                ["plum tree", "plum trees", "plum blossom", "李树", "梅树"],
                {},
            ),
            ("aspen", "aspen", False, ["aspen", "aspens", "白杨"], {"aspen": ["aspen, colorado"]}),
            ("sycamore", "sycamore", False, ["sycamore", "sycamores", "悬铃木"], {}),
        ],
    ),
    "holiday": (
        q("which holiday is the user preparing for?"),
        [
            "Write a short invitation for our celebration.",
            "Write a short note to my neighbors about our plans.",
        ],
        [
            {
                "concept": "new_year",
                "members": ["New Year", "New Year's Eve", "New Year's Day"],
                "note": "New Year's Eve merged into New Year",
            }
        ],
        [
            ("halloween", "Halloween", True, ["Halloween", "万圣节", "万圣夜"], {}),
            ("thanksgiving", "Thanksgiving", True, ["Thanksgiving", "感恩节"], {}),
            ("christmas", "Christmas", True, ["Christmas", "Xmas", "圣诞节", "圣诞"], {}),
            ("easter", "Easter", True, ["Easter", "复活节"], {"Easter": ["easter island"]}),
            (
                "new_year",
                "New Year",
                True,
                [
                    "New Year",
                    "New Year's",
                    "New Years",
                    "New Year's Eve",
                    "New Year's Day",
                    "新年",
                    "跨年",
                    "元旦",
                ],
                {
                    "New Year": ["lunar new year", "chinese new year", "jewish new year"],
                    "New Year's": ["lunar new year's", "chinese new year's"],
                    "New Years": ["lunar new years", "chinese new years"],
                    "新年": ["农历新年", "中国新年"],
                },
            ),
            (
                "valentines",
                "Valentine's Day",
                True,
                ["Valentine's Day", "Valentine", "Valentines", "情人节"],
                {},
            ),
            (
                "independence_day",
                "Independence Day",
                True,
                [
                    "Independence Day",
                    "Fourth of July",
                    "4th of July",
                    "July 4th",
                    "July Fourth",
                    "独立日",
                ],
                {},
            ),
            ("diwali", "Diwali", True, ["Diwali", "Deepavali", "排灯节"], {}),
            (
                "mothers_day",
                "Mother's Day",
                False,
                ["Mother's Day", "Mothers Day", "Mothers' Day", "母亲节"],
                {},
            ),
            (
                "lunar_new_year",
                "Lunar New Year",
                False,
                ["Lunar New Year", "Chinese New Year", "Spring Festival", "春节", "农历新年"],
                {},
            ),
            ("passover", "Passover", False, ["Passover", "Pesach", "逾越节"], {}),
        ],
    ),
}

# token-only extras (single tokens that name the concept but are not safe text aliases)
TOKEN_EXTRA: dict[str, dict[str, list[str]]] = {
    "tree": {
        "palm": [" palm", " palms"],
        "cherry": [" cherry"],
        "peach": [" peach"],
        "plum": [" plum"],
    },
    "country": {"usa": [" US"]},
}
# single-token forms judged too ambiguous for a top-10 token bag
TOKEN_DROP: set[tuple[str, str]] = {
    ("emotion", " rage"),
    ("emotion", "rage"),
    ("occupation", " author"),
    ("occupation", "author"),
    ("occupation", " developer"),
    ("occupation", "developer"),
    ("occupation", " developers"),
    ("occupation", "developers"),
}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def single(s: str) -> bool:
        return len(tok.encode(s, add_special_tokens=False)) == 1

    doc: dict = {
        "version": "sibling-concepts-v1",
        "provenance": (
            "scripts/sibling_probe/build_concepts.py; concept choice from "
            "scripts/sibling_probe/count_siblings.py (outputs/sibling/sibling_counts.json); "
            "token aliases verified against the Qwen/Qwen3.6-27B tokenizer"
        ),
        "categories": {},
    }
    for cat, (question, requests, merges, rows) in INVENTORY.items():
        concepts = []
        for cid, label, gold, aliases, exclude in rows:
            excl = {a: list(exclude.get(a, [])) for a in aliases if C.is_cjk(a)}
            excl.update({a: list(v) for a, v in exclude.items()})
            cands: list[str] = []
            for a in aliases:
                if C.is_cjk(a):
                    cands.append(a)
                else:
                    # the bare (no-space) form is usually a subword fragment ("hen" in
                    # "Stephen"); keep it only for capitalised names of 4+ characters
                    cands += [" " + a] + ([a] if a[0].isupper() and len(a) >= 4 else [])
            cands += TOKEN_EXTRA.get(cat, {}).get(cid, [])
            toks = []
            for t in cands:
                if t not in toks and (cat, t) not in TOKEN_DROP and single(t):
                    toks.append(t)
            concepts.append(
                {
                    "id": cid,
                    "label": label,
                    "gold": gold,
                    "aliases": aliases,
                    "exclude": {a: excl[a] for a in aliases if a in excl},
                    "token_aliases": toks,
                    "token_eligible": bool(toks),
                }
            )
        doc["categories"][cat] = {
            "question": question,
            "requests": requests,
            "merges": merges,
            "concepts": concepts,
        }
    cats = C.parse_inventory(doc)
    errs = C.validate_inventory(cats)
    for c in cats.values():
        errs += C.overlap_problems(c)
    if errs:
        raise SystemExit("inventory problems:\n  " + "\n  ".join(errs))
    C.CONCEPTS_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    for cat, c in cats.items():
        inel = [x.id for x in c.concepts if not x.token_eligible]
        print(f"{cat:13s} {len(c.concepts):2d} concepts; not token-eligible: {inel}")
    print(f"-> {C.CONCEPTS_PATH}")


if __name__ == "__main__":
    main()
