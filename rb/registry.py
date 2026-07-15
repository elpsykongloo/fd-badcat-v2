# -*- coding: utf-8 -*-
"""rb/registry.py — RB v2 domains, tools, and scenario blueprints
(docs/rb_design.md v2 §3). Four self-built domains borrowing the FDB domain
TYPOLOGY with fully original tools and scenarios (zero item overlap with
FDB's mock_apis — none of these tool names exist there).

Tool record: kappa in {READ, REV, COMP, IRR}; reverse = compensation tool
(None for READ/IRR); latency class in {short, mid, heavy} (lognormal params in
rb/sandbox.py); idempotency keys are always carried by the sandbox.

Scenario blueprint: ordered steps (args reference {slot} values and $R<i> =
the id returned by step i — the sandbox mints DETERMINISTIC ids, so gold calls
and the gold end-state are fully precomputable at build time); `revisable`
lists slots a revision may retarget; utterances are bilingual templates.
"""

KAPPA_ORDER = ("READ", "REV", "COMP", "IRR")

# Spoken -> canonical value maps (rb_v2.2): utterances speak the SPOKEN form,
# gold args carry the CANONICAL form the catalog declares — the dev smoke
# showed the decider (correctly) canonicalizes what it hears (800, USD, 2),
# so verbatim-spoken gold was a construction-validity bug, not model error.
CANON = {
    "amount": {"三千": 3000, "五千": 5000, "八百": 800,
               "three thousand": 3000, "five thousand": 5000,
               "eight hundred": 800},
    "threshold": {"五百": 500, "一千": 1000,
                  "five hundred": 500, "one thousand": 1000},
    "max_rent": {"六千": 6000, "八千": 8000,
                 "two thousand": 2000, "three thousand": 3000},
    "qty": {"一": 1, "两": 2, "one": 1, "two": 2},
    "nights": {"两": 2, "三": 3, "two": 2, "three": 3},
    "beds": {"一居": 1, "两居": 2, "one-bedroom": 1, "two-bedroom": 2},
    "from_cur": {"人民币": "CNY", "dollars": "USD"},
    "to_cur": {"美元": "USD", "日元": "JPY", "euros": "EUR", "yen": "JPY"},
    "date": {"五月三号": "5月3日", "五月八号": "5月8日", "六月一号": "6月1日",
             "May third": "May 3", "May eighth": "May 8", "June first": "June 1"},
    "checkin": {"五月三号": "5月3日", "五月十号": "5月10日",
                "May third": "May 3", "May tenth": "May 10"},
}

# Rendered into the tool catalog so the canonical form is DECLARED, not guessed.
ARG_FORMAT = {
    "amount": "integer", "threshold": "integer", "max_rent": "integer",
    "qty": "integer", "nights": "integer", "beds": "integer",
    "from_cur": "ISO code, e.g. USD/CNY", "to_cur": "ISO code, e.g. EUR/JPY",
    "date": "e.g. May 3 / 5月3日", "checkin": "e.g. May 3 / 5月3日",
    "item_id": "e.g. A100", "listing_id": "e.g. LST12",
}


def canon_value(slot, spoken):
    return CANON.get(slot, {}).get(spoken, spoken)

TOOLS = {
    # ecommerce
    "search_catalog":   {"domain": "ecommerce", "kappa": "READ", "required": ["query"], "reverse": None, "latency": "short"},
    "check_stock":      {"domain": "ecommerce", "kappa": "READ", "required": ["item_id"], "reverse": None, "latency": "short"},
    "add_item":         {"domain": "ecommerce", "kappa": "REV",  "required": ["item_id", "qty"], "reverse": "remove_item", "latency": "short"},
    "remove_item":      {"domain": "ecommerce", "kappa": "REV",  "required": ["item_id"], "reverse": None, "latency": "short"},
    "place_order":      {"domain": "ecommerce", "kappa": "COMP", "required": ["cart_id", "address"], "reverse": "cancel_order", "latency": "mid"},
    "cancel_order":     {"domain": "ecommerce", "kappa": "REV",  "required": ["order_id"], "reverse": None, "latency": "short"},
    # finance
    "get_balance":      {"domain": "finance", "kappa": "READ", "required": ["account"], "reverse": None, "latency": "short"},
    "get_fx_quote":     {"domain": "finance", "kappa": "READ", "required": ["amount", "from_cur", "to_cur"], "reverse": None, "latency": "short"},
    "set_alert":        {"domain": "finance", "kappa": "REV",  "required": ["threshold", "account"], "reverse": "clear_alert", "latency": "short"},
    "clear_alert":      {"domain": "finance", "kappa": "REV",  "required": ["alert_id"], "reverse": None, "latency": "short"},
    "schedule_payment": {"domain": "finance", "kappa": "REV",  "required": ["payee", "amount", "date"], "reverse": "unschedule_payment", "latency": "mid"},
    "unschedule_payment": {"domain": "finance", "kappa": "REV", "required": ["payment_id"], "reverse": None, "latency": "short"},
    "transfer_funds":   {"domain": "finance", "kappa": "COMP", "required": ["from_acct", "to_acct", "amount"], "reverse": "reverse_transfer", "latency": "mid"},
    "reverse_transfer": {"domain": "finance", "kappa": "COMP", "required": ["transfer_id"], "reverse": None, "latency": "mid"},
    # housing
    "search_rentals":   {"domain": "housing", "kappa": "READ", "required": ["city", "beds", "max_rent"], "reverse": None, "latency": "short"},
    "get_commute_time": {"domain": "housing", "kappa": "READ", "required": ["from_addr", "to_addr"], "reverse": None, "latency": "short"},
    "save_listing":     {"domain": "housing", "kappa": "REV",  "required": ["listing_id"], "reverse": "unsave_listing", "latency": "short"},
    "unsave_listing":   {"domain": "housing", "kappa": "REV",  "required": ["listing_id"], "reverse": None, "latency": "short"},
    "book_viewing":     {"domain": "housing", "kappa": "COMP", "required": ["listing_id", "slot"], "reverse": "cancel_viewing", "latency": "mid"},
    "cancel_viewing":   {"domain": "housing", "kappa": "REV",  "required": ["viewing_id"], "reverse": None, "latency": "short"},
    "submit_application": {"domain": "housing", "kappa": "IRR", "required": ["listing_id", "applicant"], "reverse": None, "latency": "mid"},
    # travel
    "search_trains":    {"domain": "travel", "kappa": "READ", "required": ["origin", "destination", "date"], "reverse": None, "latency": "short"},
    "check_visa_rule":  {"domain": "travel", "kappa": "READ", "required": ["country", "nationality"], "reverse": None, "latency": "short"},
    "hold_seat":        {"domain": "travel", "kappa": "REV",  "required": ["train_id", "seat_class"], "reverse": "release_seat", "latency": "short"},
    "release_seat":     {"domain": "travel", "kappa": "REV",  "required": ["hold_id"], "reverse": None, "latency": "short"},
    "reserve_hotel":    {"domain": "travel", "kappa": "COMP", "required": ["city", "checkin", "nights"], "reverse": "cancel_hotel", "latency": "mid"},
    "cancel_hotel":     {"domain": "travel", "kappa": "REV",  "required": ["booking_id"], "reverse": None, "latency": "short"},
    "purchase_ticket":  {"domain": "travel", "kappa": "IRR",  "required": ["hold_id", "passenger"], "reverse": None, "latency": "mid"},
}

DOMAINS = ("ecommerce", "finance", "housing", "travel")

SLOT_POOLS = {
    "zh": {"origin": ["北京", "上海", "广州"], "destination": ["杭州", "成都", "西安", "南京"],
           "date": ["五月三号", "五月八号", "六月一号"], "seat_class": ["二等座", "一等座", "商务座"],
           "passenger": ["王磊", "李娜", "张伟"], "city": ["杭州", "成都", "厦门", "青岛"],
           "checkin": ["五月三号", "五月十号"], "nights": ["两", "三"],
           "query": ["蓝牙耳机", "保温杯", "跑步鞋"], "qty": ["一", "两"],
           "address": ["海淀区学院路一号", "浦东新区世纪大道八号"],
           "payee": ["房东", "物业"], "amount": ["三千", "五千", "八百"],
           "from_cur": ["人民币"], "to_cur": ["美元", "日元"],
           "from_acct": ["工资卡"], "to_acct": ["房租卡", "储蓄卡"],
           "account": ["工资卡"], "threshold": ["五百", "一千"],
           "beds": ["一居", "两居"], "max_rent": ["六千", "八千"],
           "from_addr": ["中关村", "望京"], "to_addr": ["国贸", "西二旗"],
           "slot": ["周六上午", "周日下午"], "applicant": ["王磊", "李娜"],
           "country": ["日本", "泰国"], "nationality": ["中国"],
           "item_id": ["A100", "B205", "C330"], "listing_id": ["LST12", "LST47"]},
    "en": {"origin": ["Boston", "Seattle", "Chicago"], "destination": ["Austin", "Denver", "Portland", "Atlanta"],
           "date": ["May third", "May eighth", "June first"], "seat_class": ["coach", "first class", "business"],
           "passenger": ["Alex Chen", "Jordan Lee", "Sam Rivera"], "city": ["Austin", "Denver", "Miami", "Boise"],
           "checkin": ["May third", "May tenth"], "nights": ["two", "three"],
           "query": ["wireless earbuds", "thermos bottle", "running shoes"], "qty": ["one", "two"],
           "address": ["12 College Road", "88 Century Avenue"],
           "payee": ["the landlord", "the property office"], "amount": ["three thousand", "five thousand", "eight hundred"],
           "from_cur": ["dollars"], "to_cur": ["euros", "yen"],
           "from_acct": ["checking"], "to_acct": ["rent account", "savings"],
           "account": ["checking"], "threshold": ["five hundred", "one thousand"],
           "beds": ["one-bedroom", "two-bedroom"], "max_rent": ["two thousand", "three thousand"],
           "from_addr": ["Midtown", "the university"], "to_addr": ["Downtown", "the tech park"],
           "slot": ["Saturday morning", "Sunday afternoon"], "applicant": ["Alex Chen", "Jordan Lee"],
           "country": ["Japan", "Thailand"], "nationality": ["American"],
           "item_id": ["A100", "B205", "C330"], "listing_id": ["LST12", "LST47"]},
}

SCENARIOS = {
    "travel_chain": {
        "domain": "travel", "kind": "chain",
        "steps": [{"fn": "search_trains", "args": {"origin": "{origin}", "destination": "{destination}", "date": "{date}"}},
                  {"fn": "hold_seat", "args": {"train_id": "$R0", "seat_class": "{seat_class}"}},
                  {"fn": "purchase_ticket", "args": {"hold_id": "$R1", "passenger": "{passenger}"}}],
        "revisable": ["destination", "seat_class", "date"],
        "utt": {"zh": "帮我查{date}从{origin}到{destination}的火车，订{seat_class}，然后直接买票，乘客是{passenger}。",
                "en": "Find a train from {origin} to {destination} on {date}, hold a {seat_class} seat, and buy the ticket for {passenger}."}},
    "travel_hotel": {
        "domain": "travel", "kind": "single",
        "steps": [{"fn": "reserve_hotel", "args": {"city": "{city}", "checkin": "{checkin}", "nights": "{nights}"}}],
        "revisable": ["city", "checkin", "nights"],
        "utt": {"zh": "帮我在{city}订一间酒店，{checkin}入住，住{nights}晚。",
                "en": "Book me a hotel in {city}, checking in {checkin}, for {nights} nights."}},
    "travel_multi": {
        "domain": "travel", "kind": "multi",
        "steps": [{"fn": "search_trains", "args": {"origin": "{origin}", "destination": "{destination}", "date": "{date}"}},
                  {"fn": "check_visa_rule", "args": {"country": "{country}", "nationality": "{nationality}"}}],
        "revisable": ["destination", "country"],
        "utt": {"zh": "查一下{date}从{origin}到{destination}的火车，顺便查{nationality}去{country}要不要签证。",
                "en": "Look up trains from {origin} to {destination} on {date}, and also check if {nationality} citizens need a visa for {country}."}},
    "ecom_chain": {
        "domain": "ecommerce", "kind": "chain",
        "steps": [{"fn": "search_catalog", "args": {"query": "{query}"}},
                  {"fn": "add_item", "args": {"item_id": "$R0", "qty": "{qty}"}},
                  {"fn": "place_order", "args": {"cart_id": "$R1", "address": "{address}"}}],
        "revisable": ["query", "qty", "address"],
        "utt": {"zh": "帮我搜{query}，加{qty}件进购物车，下单寄到{address}。",
                "en": "Search for {query}, add {qty} to the cart, and order it to {address}."}},
    "ecom_single": {
        "domain": "ecommerce", "kind": "single",
        "steps": [{"fn": "add_item", "args": {"item_id": "{item_id}", "qty": "{qty}"}}],
        "revisable": ["item_id", "qty"],
        "utt": {"zh": "把编号{item_id}的商品加{qty}件到购物车。",
                "en": "Add {qty} of item {item_id} to my cart."}},
    "ecom_multi": {
        "domain": "ecommerce", "kind": "multi",
        "steps": [{"fn": "search_catalog", "args": {"query": "{query}"}},
                  {"fn": "check_stock", "args": {"item_id": "{item_id}"}}],
        "revisable": ["query", "item_id"],
        "utt": {"zh": "搜一下{query}，再看看编号{item_id}还有没有货。",
                "en": "Search for {query}, and check whether item {item_id} is in stock."}},
    "fin_transfer": {
        "domain": "finance", "kind": "chain",
        "steps": [{"fn": "get_fx_quote", "args": {"amount": "{amount}", "from_cur": "{from_cur}", "to_cur": "{to_cur}"}},
                  {"fn": "transfer_funds", "args": {"from_acct": "{from_acct}", "to_acct": "{to_acct}", "amount": "{amount}"}}],
        "revisable": ["amount", "to_acct", "to_cur"],
        "utt": {"zh": "先查{amount}块{from_cur}换{to_cur}的汇率，然后从{from_acct}转{amount}到{to_acct}。",
                "en": "Get a quote for {amount} {from_cur} to {to_cur}, then transfer {amount} from {from_acct} to {to_acct}."}},
    "fin_payment": {
        "domain": "finance", "kind": "single",
        "steps": [{"fn": "schedule_payment", "args": {"payee": "{payee}", "amount": "{amount}", "date": "{date}"}}],
        "revisable": ["amount", "date", "payee"],
        "utt": {"zh": "帮我设一笔给{payee}的付款，{amount}块，{date}扣。",
                "en": "Schedule a payment of {amount} to {payee} on {date}."}},
    "fin_multi": {
        "domain": "finance", "kind": "multi",
        "steps": [{"fn": "get_balance", "args": {"account": "{account}"}},
                  {"fn": "set_alert", "args": {"threshold": "{threshold}", "account": "{account}"}}],
        "revisable": ["threshold"],
        "utt": {"zh": "看下{account}的余额，再设一个低于{threshold}就提醒我的警报。",
                "en": "Check my {account} balance, and set an alert if it drops below {threshold}."}},
    "hou_chain": {
        "domain": "housing", "kind": "chain",
        "steps": [{"fn": "search_rentals", "args": {"city": "{city}", "beds": "{beds}", "max_rent": "{max_rent}"}},
                  {"fn": "book_viewing", "args": {"listing_id": "$R0", "slot": "{slot}"}}],
        "revisable": ["city", "beds", "max_rent", "slot"],
        "utt": {"zh": "在{city}找{beds}、租金不超过{max_rent}的房子，约{slot}看房。",
                "en": "Find a {beds} in {city} under {max_rent}, and book a viewing for {slot}."}},
    "hou_single": {
        "domain": "housing", "kind": "single",
        "steps": [{"fn": "save_listing", "args": {"listing_id": "{listing_id}"}}],
        "revisable": ["listing_id"],
        "utt": {"zh": "把编号{listing_id}的房源收藏一下。",
                "en": "Save listing {listing_id} for me."}},
    "hou_multi": {
        "domain": "housing", "kind": "multi",
        "steps": [{"fn": "search_rentals", "args": {"city": "{city}", "beds": "{beds}", "max_rent": "{max_rent}"}},
                  {"fn": "get_commute_time", "args": {"from_addr": "{from_addr}", "to_addr": "{to_addr}"}}],
        "revisable": ["city", "to_addr"],
        "utt": {"zh": "在{city}找{beds}、不超过{max_rent}的房子，再算下从{from_addr}到{to_addr}的通勤时间。",
                "en": "Find a {beds} in {city} under {max_rent}, and check the commute from {from_addr} to {to_addr}."}},
}

SCENARIOS_BY_KIND = {
    k: [sid for sid, s in SCENARIOS.items() if s["kind"] == k]
    for k in ("chain", "single", "multi")}


def worst_kappa_of(scn_id):
    ks = [TOOLS[st["fn"]]["kappa"] for st in SCENARIOS[scn_id]["steps"]]
    return max(ks, key=KAPPA_ORDER.index)
