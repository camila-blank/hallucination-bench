import json,collections,re,sys
meta={m['id']:m for m in json.load(open('meta.json'))}
specs={s['id']:s for s in json.load(open('specs.json'))}
V={
"emotion":{"embarrassed":["embarrass","humiliat","mortif","尴尬"],"angry":["angry","anger","furious","rage","frustrat","愤怒","生气"],"sad":["sad","grief","griev","sorrow","heartbr","悲伤","难过"],"nervous":["nervous","anxi","worried","焦虑","紧张"],"jealous":["jealous","envy","envious","嫉妒"],"proud":["proud","pride","骄傲","自豪"],"happy":["happy","joy","excite","开心","快乐"],"afraid":["afraid","fear","scared","害怕"],"guilty":["guilt","ashamed","shame","内疚"],"lonely":["lonely","loneli","孤独"],"disappointed":["disappoint","失望"],"grateful":["grateful","gratitude","感激"],"relieved":["relie"]},
"farm_animal":{"chicken":["chicken","hen","rooster","鸡"],"cow":["cow","cattle","dairy","牛"],"pig":["pig","sow","hog","猪"],"sheep":["sheep","lamb","ewe","羊"],"goat":["goat","山羊"],"horse":["horse","mare","stallion","马"],"duck":["duck","drake","鸭"],"turkey":["turkey","火鸡"],"rabbit":["rabbit","兔"],"dog":["dog","狗"],"cat":["cat","猫"]},
"capital_city":{"paris":["paris","巴黎"],"london":["london","伦敦"],"tokyo":["tokyo","东京"],"rome":["rome","罗马"],"berlin":["berlin","柏林"],"cairo":["cairo","开罗"],"washington":["washington","华盛顿"],"moscow":["moscow","莫斯科"],"madrid":["madrid"],"beijing":["beijing","北京"],"kyoto":["kyoto","京都"],"new_york":["new york","纽约"],"amsterdam":["amsterdam"],"vienna":["vienna"],"prague":["prague"],"barcelona":["barcelona"],"florence":["florence"],"venice":["venice"],"istanbul":["istanbul"],"seoul":["seoul"],"st_petersburg":["petersburg"]},
"tree":{"maple":["maple","枫"],"oak":["oak","acorn","橡"],"pine":["pine","松"],"birch":["birch","桦"],"willow":["willow","柳"],"palm":["palm","coconut","椰"],"cherry":["cherry","sakura","樱"],"apple":["apple","苹果"],"spruce":["spruce"],"fir":["fir tree"," fir"],"cedar":["cedar"],"elm":["elm"],"aspen":["aspen"],"magnolia":["magnolia"],"peach":["peach"],"orange":["orange tree","citrus"],"redwood":["redwood","sequoia"]},
"holiday":{"halloween":["halloween","万圣"],"thanksgiving":["thanksgiving","感恩"],"christmas":["christmas","xmas","圣诞"],"easter":["easter","复活节"],"new_year":["new year","新年"],"valentines":["valentine","情人节"],"independence_day":["fourth of july","independence day","july 4","4th of july"],"diwali":["diwali","排灯"],"hanukkah":["hanukkah","chanukah"],"lunar_new_year":["lunar new year","chinese new year","春节"],"birthday":["birthday","生日"],"st_patricks":["patrick"],"memorial_day":["memorial day"],"labor_day":["labor day"],"mothers_day":["mother's day","mothers day"],"holi":[" holi "],"eid":[" eid"]},
}
rows=[json.loads(l) for l in open('s3d.jsonl')]
def hits(text,cat):
    t=text.lower()
    out=set()
    for c,al in V[cat].items():
        for a in al:
            if re.search(r'[一-鿿]',a):
                if a in text: out.add(c)
            elif re.search(r'\b'+re.escape(a.strip()), t): out.add(c)
    return out
per=collections.defaultdict(lambda: collections.Counter()); itemsib=collections.defaultdict(set); cells=collections.Counter(); sibcells=collections.Counter()
ex=collections.defaultdict(list)
for r in rows:
    m=meta[r['id']]; cat=m['category']; g=m['gold']
    prompt=specs[r['id']]['messages'][-1]['content']
    pin=hits(prompt,cat)
    h=hits(r['samples'][0],cat) if r['samples'] else set()
    sib=h-{g}-pin
    cells[cat]+=1
    per[cat].update(sib)
    if sib: sibcells[cat]+=1; itemsib[cat].add(r['id']); ex[cat].append((r['id'],g,r['layer'],r['pos'],sorted(sib),r['samples'][0]))
    if g in h: per[cat]['GOLD:'+g]+=0
for cat in V:
    print(f"\n== {cat}: cells {cells[cat]}, sibling cells {sibcells[cat]} ({sibcells[cat]/cells[cat]:.0%}), items with sib {len(itemsib[cat])}/8")
    print("   ", per[cat].most_common(12))
json.dump(ex,open('sib_examples.json','w'),ensure_ascii=False,indent=0)
