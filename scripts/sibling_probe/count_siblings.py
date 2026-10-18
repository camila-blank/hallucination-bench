"""Count sibling mentions per category with a BROAD candidate vocabulary (inventory input).

Plan 0003 §Phase A: every sibling seen at least twice (cells) in a category's pilot readouts is
added to that category's concepts (cap 16). Sources:

* the 5 new categories: the pilot s3d-rl600 readouts ``outputs/sibling/pilot/s3d.jsonl``
  (layers 28/36/44/52, ``from_last_sentence_start``), prompts from
  ``data/sibling_probe/pilot/specs.json``;
* country / occupation: the wsbench ``user_modeling`` readouts of the selfdescribe Country /
  Occupation items for s3d-rl600 and jlens (layers 28/36/44/52; jlens = top-10 tokens, each
  decoded token matched as text) and nla-rl-L42 (L42).

A sibling mention = a candidate that is not the gold and not named in the prompt. The
candidate vocabulary below is deliberately wider than the final inventory; it uses the
package matcher (word boundaries, CJK >= 2 chars). Output: ``outputs/sibling/sibling_counts.json``
(per category: cells per candidate, per source).

    uv run python scripts/sibling_probe/count_siblings.py
"""

import json
from collections import Counter, defaultdict

from hallucination_bench import REPO_ROOT, paths
from hallucination_bench.evals.sibling_probe.concepts import Category, Concept, Matcher, is_cjk
from hallucination_bench.readouts import decode_token

PILOT = REPO_ROOT / "data" / "sibling_probe" / "pilot"
PILOT_READOUTS = REPO_ROOT / "outputs" / "sibling" / "pilot" / "s3d.jsonl"
WSB_READOUTS = paths.READOUTS_ROOT
WSB_ITEMS = paths.WSBENCH_PUB / "evals" / "user_modeling" / "items.json"
OUT = REPO_ROOT / "outputs" / "sibling" / "sibling_counts.json"
GRID = {28, 36, 44, 52}

# candidate id -> aliases (English lower/any case, CJK >= 2 chars)
BROAD: dict[str, dict[str, list[str]]] = {
    "country": {
        "paraguay": ["Paraguay", "Paraguayan", "巴拉圭"],
        "south_korea": ["South Korea", "Korea", "Korean", "Koreans", "韩国"],
        "north_korea": ["North Korea", "North Korean", "朝鲜"],
        "ukraine": ["Ukraine", "Ukrainian", "乌克兰"],
        "new_zealand": ["New Zealand", "New Zealander", "Kiwi", "新西兰"],
        "south_africa": ["South Africa", "South African", "南非"],
        "costa_rica": ["Costa Rica", "Costa Rican", "哥斯达黎加"],
        "ireland": ["Ireland", "Irish", "爱尔兰"],
        "japan": ["Japan", "Japanese", "日本"],
        "china": ["China", "Chinese", "中国"],
        "taiwan": ["Taiwan", "Taiwanese", "台湾"],
        "russia": ["Russia", "Russian", "俄罗斯"],
        "poland": ["Poland", "Polish", "波兰"],
        "belarus": ["Belarus", "Belarusian", "白俄罗斯"],
        "kazakhstan": ["Kazakhstan", "Kazakh", "哈萨克"],
        "australia": ["Australia", "Australian", "澳大利亚", "澳洲"],
        "canada": ["Canada", "Canadian", "加拿大"],
        "usa": ["United States", "USA", "U.S.", "America", "American", "美国"],
        "uk": ["United Kingdom", "UK", "Britain", "British", "England", "英国"],
        "scotland": ["Scotland", "Scottish", "苏格兰"],
        "nigeria": ["Nigeria", "Nigerian", "尼日利亚"],
        "kenya": ["Kenya", "Kenyan", "肯尼亚"],
        "zimbabwe": ["Zimbabwe", "Zimbabwean"],
        "ghana": ["Ghana", "Ghanaian"],
        "mexico": ["Mexico", "Mexican", "墨西哥"],
        "brazil": ["Brazil", "Brazilian", "巴西"],
        "argentina": ["Argentina", "Argentine", "Argentinian", "阿根廷"],
        "colombia": ["Colombia", "Colombian", "哥伦比亚"],
        "peru": ["Peru", "Peruvian", "秘鲁"],
        "chile": ["Chile", "Chilean", "智利"],
        "bolivia": ["Bolivia", "Bolivian"],
        "uruguay": ["Uruguay", "Uruguayan"],
        "venezuela": ["Venezuela", "Venezuelan"],
        "cuba": ["Cuba", "Cuban", "古巴"],
        "puerto_rico": ["Puerto Rico", "Puerto Rican"],
        "guatemala": ["Guatemala", "Guatemalan"],
        "nicaragua": ["Nicaragua", "Nicaraguan"],
        "panama": ["Panama", "Panamanian"],
        "spain": ["Spain", "Spanish", "西班牙"],
        "portugal": ["Portugal", "Portuguese", "葡萄牙"],
        "france": ["France", "French", "法国"],
        "italy": ["Italy", "Italian", "意大利"],
        "germany": ["Germany", "German", "德国"],
        "greece": ["Greece", "Greek", "希腊"],
        "turkey": ["Turkey", "Turkish", "土耳其"],
        "india": ["India", "Indian", "印度"],
        "pakistan": ["Pakistan", "Pakistani"],
        "vietnam": ["Vietnam", "Vietnamese", "越南"],
        "thailand": ["Thailand", "Thai", "泰国"],
        "philippines": ["Philippines", "Filipino", "菲律宾"],
        "indonesia": ["Indonesia", "Indonesian", "印度尼西亚", "印尼"],
        "egypt": ["Egypt", "Egyptian", "埃及"],
        "morocco": ["Morocco", "Moroccan"],
        "iran": ["Iran", "Iranian", "伊朗"],
        "israel": ["Israel", "Israeli", "以色列"],
        "jamaica": ["Jamaica", "Jamaican"],
        "netherlands": ["Netherlands", "Dutch", "荷兰"],
        "sweden": ["Sweden", "Swedish", "瑞典"],
        "norway": ["Norway", "Norwegian", "挪威"],
        "hungary": ["Hungary", "Hungarian"],
        "romania": ["Romania", "Romanian"],
        "czech": ["Czech", "Czechia"],
        "georgia": ["Georgia", "Georgian"],
        "serbia": ["Serbia", "Serbian"],
        "lithuania": ["Lithuania", "Lithuanian"],
        "fiji": ["Fiji", "Fijian"],
        "samoa": ["Samoa", "Samoan"],
        "hawaii": ["Hawaii", "Hawaiian"],
    },
    "occupation": {
        "fisherman": ["fisherman", "fishermen", "fisher", "渔民", "渔夫"],
        "locksmith": ["locksmith", "locksmiths", "锁匠"],
        "flight_attendant": ["flight attendant", "stewardess", "cabin crew", "空乘", "空姐"],
        "anthropologist": ["anthropologist", "人类学家"],
        "farmer": ["farmer", "farmers", "农民"],
        "sailor": ["sailor", "sailors", "seaman", "水手"],
        "captain": ["captain", "船长"],
        "pilot": ["pilot", "pilots", "飞行员"],
        "plumber": ["plumber", "plumbers", "水管工"],
        "electrician": ["electrician", "电工"],
        "carpenter": ["carpenter", "木匠"],
        "mechanic": ["mechanic", "机械师"],
        "security_guard": ["security guard", "保安"],
        "police_officer": ["police officer", "policeman", "cop", "警察"],
        "firefighter": ["firefighter", "fireman", "消防员"],
        "detective": ["detective", "侦探"],
        "teacher": ["teacher", "teachers", "教师", "老师"],
        "professor": ["professor", "教授"],
        "researcher": ["researcher", "研究员"],
        "scientist": ["scientist", "科学家"],
        "archaeologist": ["archaeologist", "考古学家"],
        "sociologist": ["sociologist", "社会学家"],
        "historian": ["historian", "历史学家"],
        "ethnographer": ["ethnographer"],
        "linguist": ["linguist", "语言学家"],
        "journalist": ["journalist", "记者"],
        "writer": ["writer", "author", "作家"],
        "photographer": ["photographer", "摄影师"],
        "travel_writer": ["travel writer"],
        "nurse": ["nurse", "护士"],
        "doctor": ["doctor", "physician", "医生"],
        "paramedic": ["paramedic"],
        "waiter": ["waiter", "waitress", "服务员"],
        "chef": ["chef", "cook", "厨师"],
        "bartender": ["bartender"],
        "hotel_concierge": ["concierge"],
        "tour_guide": ["tour guide", "导游"],
        "customer_service": ["customer service"],
        "air_traffic_controller": ["air traffic controller"],
        "travel_agent": ["travel agent"],
        "lawyer": ["lawyer", "attorney", "律师"],
        "engineer": ["engineer", "工程师"],
        "software_engineer": ["software engineer", "programmer", "developer", "程序员"],
        "hacker": ["hacker", "黑客"],
        "thief": ["thief", "burglar", "小偷"],
        "handyman": ["handyman"],
        "janitor": ["janitor", "custodian"],
        "blacksmith": ["blacksmith", "铁匠"],
        "jeweler": ["jeweler", "jeweller"],
        "watchmaker": ["watchmaker", "clockmaker"],
        "marine_biologist": ["marine biologist"],
        "oceanographer": ["oceanographer"],
        "meteorologist": ["meteorologist", "weather forecaster", "气象学家"],
        "boat_builder": ["boatbuilder", "boat builder", "shipwright"],
        "lighthouse_keeper": ["lighthouse keeper"],
        "lifeguard": ["lifeguard"],
        "diver": ["diver", "潜水员"],
        "shepherd": ["shepherd"],
        "hunter": ["hunter", "猎人"],
        "soldier": ["soldier", "士兵"],
        "artist": ["artist", "艺术家"],
        "musician": ["musician", "音乐家"],
        "actor": ["actor", "actress", "演员"],
        "student": ["student", "学生"],
        "manager": ["manager", "经理"],
        "entrepreneur": ["entrepreneur", "企业家"],
        "sales": ["salesman", "salesperson"],
        "accountant": ["accountant", "会计"],
    },
    "emotion": {
        "embarrassed": [
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
        ],
        "angry": [
            "angry",
            "anger",
            "furious",
            "fury",
            "frustrated",
            "frustration",
            "frustrating",
            "outraged",
            "irritated",
            "annoyed",
            "rage",
            "愤怒",
            "生气",
            "恼火",
        ],
        "sad": [
            "sad",
            "sadness",
            "grief",
            "grieving",
            "heartbroken",
            "heartbreak",
            "sorrow",
            "mourning",
            "悲伤",
            "难过",
            "伤心",
        ],
        "nervous": [
            "nervous",
            "nervousness",
            "anxious",
            "anxiety",
            "worried",
            "worry",
            "焦虑",
            "紧张",
            "担心",
        ],
        "jealous": ["jealous", "jealousy", "envy", "envious", "嫉妒", "羡慕"],
        "proud": ["proud", "pride", "骄傲", "自豪"],
        "happy": ["happy", "happiness", "joy", "joyful", "delighted", "开心", "快乐", "高兴"],
        "excited": ["excited", "excitement", "thrilled", "兴奋", "激动"],
        "afraid": ["afraid", "fear", "scared", "terrified", "frightened", "害怕", "恐惧"],
        "guilty": ["guilty", "guilt", "内疚"],
        "lonely": ["lonely", "loneliness", "孤独"],
        "disappointed": ["disappointed", "disappointment", "失望"],
        "grateful": ["grateful", "gratitude", "thankful", "感激", "感恩"],
        "relieved": ["relieved", "relief", "如释重负"],
        "betrayed": ["betrayed", "betrayal", "背叛"],
        "hurt": ["hurt"],
        "resentful": ["resentful", "resentment", "bitter", "bitterness"],
        "confused": ["confused", "confusion", "困惑"],
        "overwhelmed": ["overwhelmed"],
        "hopeful": ["hopeful", "hope", "希望"],
        "nostalgic": ["nostalgic", "nostalgia", "怀旧"],
        "surprised": ["surprised", "surprise", "shocked", "惊讶"],
        "insecure": ["insecure", "insecurity", "inadequate", "inadequacy"],
        "admiration": ["admiration", "admire"],
        "empathy": ["empathy", "sympathy", "compassion"],
        "love": ["love", "loving", "爱"],
        "calm": ["calm", "peaceful", "平静"],
    },
    "farm_animal": {
        "chicken": [
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
        "cow": [
            "cow",
            "cows",
            "cattle",
            "dairy cow",
            "heifer",
            "calf",
            "calves",
            "奶牛",
            "母牛",
            "牛群",
        ],
        "pig": [
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
        "sheep": ["sheep", "lamb", "lambs", "ewe", "ewes", "ram", "rams", "羊群", "绵羊", "小羊"],
        "goat": ["goat", "goats", "kid goat", "nanny goat", "山羊"],
        "horse": [
            "horse",
            "horses",
            "mare",
            "mares",
            "stallion",
            "foal",
            "pony",
            "马匹",
            "骏马",
            "母马",
        ],
        "duck": ["duck", "ducks", "duckling", "ducklings", "drake", "drakes", "鸭子", "小鸭"],
        "turkey": ["turkey", "turkeys", "火鸡"],
        "goose": ["goose", "geese", "鹅群"],
        "rabbit": ["rabbit", "rabbits", "bunny", "bunnies", "兔子"],
        "donkey": ["donkey", "donkeys", "驴子"],
        "llama": ["llama", "llamas", "alpaca", "alpacas", "羊驼"],
        "ox": ["ox", "oxen"],
        "bull": ["bull", "bulls", "公牛"],
        "dog": ["dog", "dogs", "puppy", "sheepdog", "狗"],
        "cat": ["cat", "cats", "kitten", "猫咪"],
        "bee": ["bee", "bees", "beehive", "蜜蜂"],
        "fish": ["fish"],
    },
    "capital_city": {
        "paris": ["Paris", "Parisian", "巴黎"],
        "london": ["London", "伦敦"],
        "tokyo": ["Tokyo", "东京"],
        "rome": ["Rome", "Roman", "罗马"],
        "berlin": ["Berlin", "柏林"],
        "cairo": ["Cairo", "开罗"],
        "washington": ["Washington", "D.C.", "华盛顿"],
        "moscow": ["Moscow", "莫斯科"],
        "madrid": ["Madrid", "马德里"],
        "beijing": ["Beijing", "北京"],
        "kyoto": ["Kyoto", "京都"],
        "osaka": ["Osaka", "大阪"],
        "new_york": ["New York", "NYC", "纽约"],
        "amsterdam": ["Amsterdam", "阿姆斯特丹"],
        "vienna": ["Vienna", "维也纳"],
        "prague": ["Prague", "布拉格"],
        "barcelona": ["Barcelona", "巴塞罗那"],
        "florence": ["Florence", "佛罗伦萨"],
        "venice": ["Venice", "威尼斯"],
        "milan": ["Milan", "米兰"],
        "athens": ["Athens", "雅典"],
        "istanbul": ["Istanbul", "伊斯坦布尔"],
        "seoul": ["Seoul", "首尔"],
        "st_petersburg": ["St. Petersburg", "Saint Petersburg", "圣彼得堡"],
        "lisbon": ["Lisbon", "里斯本"],
        "dublin": ["Dublin", "都柏林"],
        "edinburgh": ["Edinburgh", "爱丁堡"],
        "munich": ["Munich", "慕尼黑"],
        "budapest": ["Budapest", "布达佩斯"],
        "bangkok": ["Bangkok", "曼谷"],
        "dubai": ["Dubai", "迪拜"],
        "los_angeles": ["Los Angeles", "洛杉矶"],
        "chicago": ["Chicago", "芝加哥"],
        "boston": ["Boston", "波士顿"],
        "san_francisco": ["San Francisco", "旧金山"],
        "philadelphia": ["Philadelphia", "费城"],
        "sydney": ["Sydney", "悉尼"],
        "jerusalem": ["Jerusalem", "耶路撒冷"],
        "delhi": ["Delhi", "德里"],
        "mexico_city": ["Mexico City", "墨西哥城"],
        "luxor": ["Luxor"],
        "giza": ["Giza"],
    },
    "tree": {
        "maple": ["maple", "maples", "枫树", "枫叶"],
        "oak": ["oak", "oaks", "acorn", "acorns", "橡树", "橡子"],
        "pine": ["pine", "pines", "pinecone", "pinecones", "pine cone", "松树", "松果"],
        "birch": ["birch", "birches", "桦树", "白桦"],
        "willow": ["willow", "willows", "柳树", "垂柳"],
        "palm": ["palm tree", "palm trees", "palms", "coconut palm", "棕榈", "椰子树"],
        "cherry": ["cherry", "cherries", "cherry blossom", "sakura", "樱花", "樱桃"],
        "apple": ["apple", "apples", "苹果"],
        "spruce": ["spruce", "云杉"],
        "fir": ["fir", "firs", "冷杉"],
        "cedar": ["cedar", "雪松"],
        "elm": ["elm", "榆树"],
        "aspen": ["aspen"],
        "magnolia": ["magnolia", "玉兰"],
        "peach": ["peach", "peaches", "桃树", "桃花"],
        "plum": ["plum", "梅花", "李树"],
        "orange": ["orange tree", "citrus", "橙树"],
        "lemon": ["lemon", "柠檬"],
        "redwood": ["redwood", "sequoia", "红杉"],
        "cypress": ["cypress", "柏树"],
        "eucalyptus": ["eucalyptus", "桉树"],
        "beech": ["beech", "山毛榉"],
        "ash": ["ash tree"],
        "sycamore": ["sycamore"],
        "chestnut": ["chestnut", "栗子"],
        "walnut": ["walnut", "核桃"],
        "pear": ["pear", "梨树"],
        "olive": ["olive", "橄榄"],
        "dogwood": ["dogwood"],
        "evergreen": ["evergreen", "常青"],
        "bamboo": ["bamboo", "竹子"],
        "banyan": ["banyan", "榕树"],
        "ginkgo": ["ginkgo", "银杏"],
        "baobab": ["baobab"],
        "cottonwood": ["cottonwood"],
        "hemlock": ["hemlock"],
        "larch": ["larch"],
    },
    "holiday": {
        "halloween": ["Halloween", "万圣节"],
        "thanksgiving": ["Thanksgiving", "感恩节"],
        "christmas": ["Christmas", "Xmas", "圣诞"],
        "easter": ["Easter", "复活节"],
        "new_year": ["New Year", "New Year's", "新年", "跨年"],
        "valentines": ["Valentine", "Valentine's", "Valentines", "情人节"],
        "independence_day": [
            "Fourth of July",
            "4th of July",
            "July 4th",
            "Independence Day",
            "独立日",
        ],
        "diwali": ["Diwali", "Deepavali", "排灯节"],
        "hanukkah": ["Hanukkah", "Chanukah", "光明节"],
        "lunar_new_year": ["Lunar New Year", "Chinese New Year", "春节"],
        "st_patricks": ["St. Patrick", "Saint Patrick", "St Patrick"],
        "memorial_day": ["Memorial Day"],
        "labor_day": ["Labor Day"],
        "mothers_day": ["Mother's Day", "Mothers Day", "母亲节"],
        "fathers_day": ["Father's Day", "父亲节"],
        "holi": ["Holi"],
        "eid": ["Eid", "Ramadan", "开斋节"],
        "rosh_hashanah": ["Rosh Hashanah"],
        "passover": ["Passover"],
        "kwanzaa": ["Kwanzaa"],
        "mid_autumn": ["Mid-Autumn", "中秋"],
        "day_of_the_dead": ["Day of the Dead", "Dia de los Muertos", "Día de los Muertos"],
        "cinco_de_mayo": ["Cinco de Mayo"],
        "mardi_gras": ["Mardi Gras", "Carnival"],
        "oktoberfest": ["Oktoberfest"],
        "guy_fawkes": ["Guy Fawkes", "Bonfire Night"],
        "canada_day": ["Canada Day"],
        "bastille_day": ["Bastille Day"],
        "veterans_day": ["Veterans Day"],
        "mlk_day": ["Martin Luther King"],
        "groundhog_day": ["Groundhog Day"],
        "new_years_eve": ["New Year's Eve"],
    },
}

# exclusion phrases for the noisiest generic uses (pilot hand read)
EXCLUDE: dict[str, dict[str, list[str]]] = {
    "emotion": {
        "happy": ["happy to", "happy birthday", "happy holidays", "happy new year", "happy hour"],
        "love": ["love to", "love help", "loved one", "love for you to"],
        "grateful": ["would be grateful", "i'd be grateful"],
        "relief": ["pain relief", "comic relief"],
        "高兴": ["很高兴为", "高兴为"],
        "快乐": ["生日快乐", "新年快乐", "节日快乐"],
    },
    "country": {
        "America": ["latin america", "south america", "central america", "north america"],
        "American": ["latin american", "south american", "central american", "north american"],
        "Korea": ["north korea"],
        "Korean": ["north korean"],
    },
}

WSB_GOLD = {
    "Country": ("country", {}),
    "Occupation": ("occupation", {}),
}


def build(cat: str) -> Matcher:
    cs = []
    for cid, al in BROAD[cat].items():
        excl = {a: () for a in al if is_cjk(a)}
        excl.update({a: tuple(v) for a, v in EXCLUDE.get(cat, {}).items() if a in al})
        cs.append(Concept(cid, al[0], cat, False, tuple(al), excl))
    return Matcher(Category(cat, "", ("x",), tuple(cs)))


def main() -> None:
    out: dict = {}
    # pilot (5 new categories)
    meta = {m["id"]: m for m in json.loads((PILOT / "meta.json").read_text())}
    specs = {s["id"]: s for s in json.loads((PILOT / "specs.json").read_text())}
    per = defaultdict(lambda: defaultdict(Counter))  # cat -> source -> Counter(concept)
    items_with = defaultdict(lambda: defaultdict(set))
    ms = {c: build(c) for c in BROAD}
    for line in PILOT_READOUTS.read_text().splitlines():
        r = json.loads(line)
        m = meta[r["id"]]
        cat, gold = m["category"], m["gold"]
        mt = ms[cat]
        named = mt.mentions(specs[r["id"]]["messages"][-1]["content"])
        sib = mt.mentions(r["samples"][0] if r["samples"] else "") - {gold} - named
        per[cat]["pilot-s3d"].update(sib)
        for s in sib:
            items_with[cat][s].add(r["id"])
    # wsbench user_modeling (country / occupation)
    bank = json.loads(WSB_ITEMS.read_text())["items"]
    wsb = {
        it["name"]: it
        for it in bank
        if it["subfamily"] == "selfdescribe" and it["attr_class"] in WSB_GOLD
    }
    for arm in ("s3d-rl600", "jlens", "nla-rl-L42"):
        for line in (WSB_READOUTS / arm / "user_modeling.jsonl").open():
            r = json.loads(line)
            it = wsb.get(r["id"])
            if it is None or (arm != "nla-rl-L42" and r["layer"] not in GRID):
                continue
            cat = WSB_GOLD[it["attr_class"]][0]
            mt = ms[cat]
            gold = it["attr"]
            named = mt.mentions(it["messages"][-1]["content"])
            if arm == "jlens":
                text = "\n".join(decode_token(t) for t in r["tokens"])
            else:
                text = r["samples"][0] if r.get("samples") else ""
            sib = mt.mentions(text) - {gold} - named
            per[cat][arm].update(sib)
            for s in sib:
                items_with[cat][s].add(r["id"])
    for cat in BROAD:
        tot = Counter()
        for c in per[cat].values():
            tot.update(c)
        out[cat] = {
            "total": dict(tot.most_common()),
            "by_source": {k: dict(v.most_common()) for k, v in per[cat].items()},
            "n_items": {k: len(v) for k, v in items_with[cat].items()},
        }
        print(f"\n== {cat}")
        for cid, n in tot.most_common(30):
            src = ", ".join(f"{k}={v.get(cid, 0)}" for k, v in per[cat].items())
            print(f"  {cid:22s} {n:5d}  items={len(items_with[cat][cid]):2d}  ({src})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
