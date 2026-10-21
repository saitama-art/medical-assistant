"""知识库扩充资料清单（第一阶段）。

每条：key(文件名主干), url, title(中文名), category(知识类别), doc_type(资料类型)。
抓取与转换工具读取本清单；抓取失败的条目会在报告中列出，不冒充实测。
"""

# ---- 疾病与健康知识（默沙东诊疗手册·大众版，病人教育资料）----
MSD_DISEASES = [
    ("disease-hypertension", "/home/heart-and-blood-vessel-disorders/high-blood-pressure/high-blood-pressure", "高血压", "疾病与健康知识"),
    ("disease-coronary-artery-disease", "/home/heart-and-blood-vessel-disorders/coronary-artery-disease/overview-of-coronary-artery-disease-cad", "冠状动脉疾病（冠心病）", "疾病与健康知识"),
    ("disease-type-2-diabetes", "/home/hormonal-and-metabolic-disorders/diabetes-mellitus-and-low-blood-sugar-hypoglycemia/type-2-diabetes-mellitus", "2型糖尿病", "疾病与健康知识"),
    ("disease-asthma", "/home/lung-and-airway-disorders/asthma/asthma", "哮喘", "疾病与健康知识"),
    ("disease-copd", "/home/lung-and-airway-disorders/chronic-obstructive-pulmonary-disease-copd/chronic-obstructive-pulmonary-disease-copd", "慢性阻塞性肺疾病（COPD）", "疾病与健康知识"),
    ("disease-gastritis", "/home/digestive-disorders/gastritis-and-peptic-ulcer-disease/gastritis", "胃炎", "疾病与健康知识"),
    ("disease-osteoporosis", "/home/bone-joint-and-muscle-disorders/osteoporosis/osteoporosis", "骨质疏松症", "疾病与健康知识"),
    ("disease-iron-deficiency-anemia", "/home/quick-facts-blood-disorders/anemia/iron-deficiency-anemia", "缺铁性贫血", "疾病与健康知识"),
    ("disease-stroke", "/home/brain-spinal-cord-and-nerve-disorders/stroke/overview-of-stroke", "脑卒中（中风）", "疾病与健康知识"),
    ("disease-high-cholesterol", "/home/hormonal-and-metabolic-disorders/cholesterol-disorders/overview-of-cholesterol-and-lipid-disorders", "胆固醇及脂代谢紊乱", "疾病与健康知识"),
    ("disease-migraine", "/home/brain-spinal-cord-and-nerve-disorders/headaches/migraines", "偏头痛", "疾病与健康知识"),
    ("disease-hyperthyroidism", "/home/hormonal-and-metabolic-disorders/thyroid-gland-disorders/hyperthyroidism", "甲状腺功能亢进症", "疾病与健康知识"),
    ("disease-hypothyroidism", "/home/hormonal-and-metabolic-disorders/thyroid-gland-disorders/hypothyroidism", "甲状腺功能减退症", "疾病与健康知识"),
    # ---- 2026-10-20 第三轮高频主题 ----
    ("disease-osteoarthritis", "/home/bone-joint-and-muscle-disorders/joint-disorders/osteoarthritis-oa", "骨关节炎", "疾病与健康知识"),
    ("disease-rheumatoid-arthritis", "/home/bone-joint-and-muscle-disorders/joint-disorders/rheumatoid-arthritis-ra", "类风湿性关节炎", "疾病与健康知识"),
    ("disease-eczema", "/home/skin-disorders/itching-and-dermatitis/atopic-dermatitis-eczema", "特应性皮炎（湿疹）", "疾病与健康知识"),
    ("disease-depression", "/home/mental-health-disorders/mood-disorders/depression", "抑郁症", "疾病与健康知识"),
    ("disease-anxiety", "/home/mental-health-disorders/anxiety-and-trauma-and-stressor-related-disorders/generalized-anxiety-disorder", "广泛性焦虑障碍", "疾病与健康知识"),
    # ---- 2026-10-20 第四轮：真实验收暴露的裸词拒答缺口 ----
    # 「低血糖」裸词百科式提问曾判证据不足（片段只有药品语境内容）；该栏目独立页
    # 经验证存在（栏目索引 /home/hormonal-and-metabolic-disorders/diabetes-mellitus-
    # and-low-blood-sugar-hypoglycemia 下的 hypoglycemia 子页）。
    # 儿童/老年人用药无独立页（大众版栏目结构如此，既有"影响药物反应的因素"条目内覆盖）。
    ("disease-hypoglycemia", "/home/hormonal-and-metabolic-disorders/diabetes-mellitus-and-low-blood-sugar-hypoglycemia/hypoglycemia", "低血糖", "疾病与健康知识"),
]

# ---- 症状与就医指引（默沙东大众版·症状）----
MSD_SYMPTOMS = [
    ("symptom-chest-pain", "/home/heart-and-blood-vessel-disorders/symptoms-of-heart-and-blood-vessel-disorders/chest-pain", "胸痛", "症状与就医指引"),
    ("symptom-shortness-of-breath", "/home/lung-and-airway-disorders/symptoms-of-lung-disorders/shortness-of-breath", "气短（呼吸困难）", "症状与就医指引"),
    ("symptom-palpitations", "/home/heart-and-blood-vessel-disorders/symptoms-of-heart-and-blood-vessel-disorders/palpitations", "心悸", "症状与就医指引"),
    ("symptom-fainting", "/home/heart-and-blood-vessel-disorders/symptoms-of-heart-and-blood-vessel-disorders/fainting", "晕厥（昏厥）", "症状与就医指引"),
    ("symptom-dizziness-vertigo", "/home/ear-nose-and-throat-disorders/symptoms-of-ear-disorders/dizziness-and-vertigo", "头晕和眩晕", "症状与就医指引"),
    ("symptom-chronic-abdominal-pain", "/home/digestive-disorders/symptoms-of-digestive-disorders/chronic-abdominal-pain-and-recurring-abdominal-pain", "慢性腹痛和反复发作性腹痛", "症状与就医指引"),
    ("symptom-diarrhea-adults", "/home/digestive-disorders/symptoms-of-digestive-disorders/diarrhea-in-adults", "成人腹泻", "症状与就医指引"),
    ("symptom-cough-adults", "/home/lung-and-airway-disorders/symptoms-of-lung-disorders/cough-in-adults", "成人咳嗽", "症状与就医指引"),
    # ---- 2026-10-20 第三轮高频症状 ----
    ("symptom-nausea-vomiting", "/home/digestive-disorders/symptoms-of-digestive-disorders/nausea-and-vomiting-in-adults", "成人恶心和呕吐", "症状与就医指引"),
    ("symptom-headache-overview", "/home/brain-spinal-cord-and-nerve-disorders/headaches/overview-of-headache", "头痛概述", "症状与就医指引"),
    ("symptom-joint-pain", "/home/bone-joint-and-muscle-disorders/symptoms-of-musculoskeletal-disorders/joint-pain-many-joints", "关节疼痛（多关节）", "症状与就医指引"),
]

# ---- 检查指标解读（默沙东大众版·常见医疗检查/影像/功能检查）----
MSD_TESTS = [
    ("test-common-medical-tests", "/home/resources/common-medical-tests/common-medical-tests", "常见的医疗检查（总览）", "检查指标解读"),
    ("test-x-rays", "/home/special-subjects/common-imaging-tests/x-rays", "X线检查", "检查指标解读"),
    ("test-ct", "/home/special-subjects/common-imaging-tests/computed-tomography-ct", "计算机断层扫描（CT）", "检查指标解读"),
    ("test-mri", "/home/special-subjects/common-imaging-tests/magnetic-resonance-imaging-mri", "磁共振成像（MRI）", "检查指标解读"),
    ("test-ultrasound", "/home/special-subjects/common-imaging-tests/ultrasound", "超声检查", "检查指标解读"),
    ("test-ecg", "/home/heart-and-blood-vessel-disorders/diagnosis-of-heart-and-blood-vessel-disorders/electrocardiography", "心电图（ECG）", "检查指标解读"),
    ("test-pft", "/home/lung-and-airway-disorders/diagnosis-of-and-procedures-for-lung-disorders/pulmonary-function-testing-pft", "肺功能检查", "检查指标解读"),
    # ---- 实验室检查（2026-10-20 补：血液/肝/肾功能化验指标解读）----
    ("test-blood-lab", "/home/blood-disorders/symptoms-and-diagnosis-of-blood-disorders/laboratory-tests-for-blood-disorders", "血液疾病的实验室检查（血常规等）", "检查指标解读"),
    ("test-liver-blood", "/home/liver-and-gallbladder-disorders/diagnosis-of-liver-gallbladder-and-biliary-disorders/liver-blood-tests", "肝脏血液检测", "检查指标解读"),
    ("test-kidney-function", "/home/kidney-disorders/diagnosis-of-kidney-disorders/kidney-function-tests", "肾功能测定", "检查指标解读"),
    ("test-thyroid-overview", "/home/hormonal-and-metabolic-disorders/thyroid-gland-disorders/overview-of-the-thyroid-gland", "甲状腺概述（甲功检查）", "检查指标解读"),
]

# ---- 用药安全与特殊人群（默沙东大众版）----
MSD_MED_SAFETY = [
    ("meds-response-overview", "/home/drugs/factors-affecting-response-to-medications/overview-of-response-to-medications", "影响药物反应的因素（年龄/特殊人群）", "用药安全与特殊人群"),
    ("meds-drug-use-during-pregnancy", "/home/women-s-health-issues/drug-use-during-pregnancy/drug-use-during-pregnancy", "孕期用药", "用药安全与特殊人群"),
    # 注：大众版无独立"儿童用药/老年人用药/药物相互作用"旧路径，改收 /home/drugs 栏目的
    # 对应主题（相互作用/不良反应/用药错误/依从性），特殊人群内容在相应条目内覆盖。
    ("meds-drug-interactions", "/home/drugs/factors-affecting-response-to-medications/drug-interactions", "药物相互作用", "用药安全与特殊人群"),
    ("meds-adverse-reactions", "/home/drugs/overview-of-medications/adverse-drug-reactions", "药物不良反应", "用药安全与特殊人群"),
    ("meds-medication-errors", "/home/drugs/overview-of-medications/medication-errors", "用药错误", "用药安全与特殊人群"),
    ("meds-adherence", "/home/drugs/factors-affecting-response-to-medications/adherence-to-medication", "服药依从性", "用药安全与特殊人群"),
]

MSD_BASE = "https://www.msdmanuals.cn"

# ---- 药品说明书（中国医药信息查询平台 dayi.org.cn，经审核认证词条）----
# 名称 -> 期望抓取的详情页（在 list/4.html 列表中确认存在的直接收录；其余由 fetch 阶段搜索补全）
DAYI_KNOWN = [
    ("drug-metformin-sr", "https://www.dayi.org.cn/drug/1157161.html", "二甲双胍缓释片"),
    ("drug-rosuvastatin", "https://www.dayi.org.cn/drug/1152374.html", "瑞舒伐他汀"),
    ("drug-mecobalamin-tab", "https://www.dayi.org.cn/drug/1144944.html", "甲钴胺片"),
    # ---- 2026-10-20 第三轮：站点搜索接口解析到的常用药（列表分页只展示少量条目，
    # 原 fetch 的列表解析找不到它们，并非平台无词条）----
    ("drug-amoxicillin", "https://www.dayi.org.cn/drug/1151078.html", "阿莫西林"),
    ("drug-omeprazole", "https://www.dayi.org.cn/drug/1158996.html", "奥美拉唑胶囊"),
    ("drug-loratadine", "https://www.dayi.org.cn/drug/1151949.html", "氯雷他定"),
    ("drug-atorvastatin", "https://www.dayi.org.cn/drug/1152372.html", "阿托伐他汀"),
    ("drug-nifedipine", "https://www.dayi.org.cn/drug/1151977.html", "硝苯地平"),
    ("drug-valsartan", "https://www.dayi.org.cn/drug/1151975.html", "缬沙坦"),
    ("drug-levofloxacin", "https://www.dayi.org.cn/drug/1025034.html", "左氧氟沙星片"),
    ("drug-montmorillonite", "https://www.dayi.org.cn/drug/1025700.html", "蒙脱石散"),
    ("drug-paracetamol", "https://www.dayi.org.cn/drug/1142592.html", "对乙酰氨基酚片"),
    ("drug-cefixime", "https://www.dayi.org.cn/drug/1151919.html", "头孢克肟"),
]

# 计划内但详情页 id 待解析的常用药（fetch 阶段在平台列表/搜索中解析）。
# 注：站点列表分页只展示少量条目，解析不到不代表无词条；2026-10-20 起已知药品
# 直接收录进 DAYI_KNOWN（用 search?keyword= 接口逐个确认）。
DAYI_WANTED = [
]

# ---- 常见健康知识补缺（2026-10-19；URL 逐个 curl 验证 200 后收录）----
MSD_COMMON_HEALTH = [
    ("symptom-constipation", "/home/digestive-disorders/symptoms-of-digestive-disorders/constipation-in-adults", "成人便秘", "症状与就医指引"),
    ("symptom-insomnia", "/home/brain-spinal-cord-and-nerve-disorders/sleep-disorders/insomnia-and-excessive-daytime-sleepiness-eds", "失眠", "症状与就医指引"),
    ("symptom-fever-adults", "/home/infections/biology-of-infectious-disease/fever-in-adults", "成人发热", "症状与就医指引"),
    ("symptom-low-back-pain", "/home/bone-joint-and-muscle-disorders/low-back-and-neck-pain/low-back-pain", "腰痛", "症状与就医指引"),
    ("symptom-hives", "/home/skin-disorders/itching-and-dermatitis/hives", "荨麻疹（风团）", "症状与就医指引"),
    ("disease-gout", "/home/bone-joint-and-muscle-disorders/gout-and-calcium-pyrophosphate-arthritis/gout", "痛风（高尿酸血症）", "疾病与健康知识"),
    ("disease-h-pylori", "/home/digestive-disorders/gastritis-and-peptic-ulcer-disease/helicobacter-pylori-infection", "幽门螺杆菌感染", "疾病与健康知识"),
    ("disease-hepatitis", "/home/liver-and-gallbladder-disorders/hepatitis/overview-of-hepatitis", "肝炎概述", "疾病与健康知识"),
    ("disease-dyslipidemia", "/home/hormonal-and-metabolic-disorders/cholesterol-disorders/dyslipidemia", "血脂异常", "疾病与健康知识"),
    # ---- 2026-10-20 第五轮：高频日常健康问题（URL 逐个 curl 验证 200）----
    ("health-common-cold", "/home/infections/respiratory-virus-infections/common-cold", "普通感冒", "疾病与健康知识"),
    ("health-rhinitis", "/home/ear-nose-and-throat-disorders/nose-and-sinus-disorders/rhinitis", "鼻炎", "疾病与健康知识"),
    ("health-acne", "/home/skin-disorders/acne-and-related-disorders/acne", "痤疮（青春痘）", "疾病与健康知识"),
    ("health-hair-loss", "/home/skin-disorders/hair-disorders/alopecia-hair-loss", "脱发", "疾病与健康知识"),
    ("health-menstrual-cramps", "/home/women-s-health-issues/menstrual-disorders-and-abnormal-uterine-bleeding/menstrual-cramps", "痛经", "疾病与健康知识"),
    ("health-neck-pain", "/home/bone-joint-and-muscle-disorders/low-back-and-neck-pain/neck-pain", "颈痛", "症状与就医指引"),
    # ---- 2026-10-21 第六轮：肛肠与关节滑囊高频问题（URL 逐个 curl 验证 200）----
    ("health-hemorrhoids", "/home/digestive-disorders/anal-and-rectal-disorders/hemorrhoids", "痔疮", "疾病与健康知识"),
    ("health-anal-fissure", "/home/digestive-disorders/anal-and-rectal-disorders/anal-fissure", "肛裂", "疾病与健康知识"),
    ("health-anal-itching", "/home/digestive-disorders/anal-and-rectal-disorders/anal-itching", "肛门瘙痒", "症状与就医指引"),
    ("health-bursitis", "/home/bone-joint-and-muscle-disorders/muscle-bursa-and-tendon-disorders/bursitis", "滑囊炎（关节肿痛）", "疾病与健康知识"),
]

# ---- 饮食营养与体重管理（2026-10-21 第七轮 P0：覆盖「我想减肥，饮食如何安排」类
# 证据不足缺口；URL 逐个 curl 验证 200）。栏目定位说明：
# - 「肥胖」词条内含治疗/饮食/改变饮食习惯各节，是减重问题的核心依据；
# - 营养概述子栏目（卡路里/三大营养素/营养需求/维生素矿物质）覆盖能量估算、
#   膳食结构、蛋白质与微量营养素问题；
# - 锻炼三篇服务"运动为主还是力量为主/初学者计划/运动安全"（与 P1 运动主题共享）；
# - 慢性肾病覆盖特殊人群饮食（低蛋白/限盐）；膳食补充剂概述覆盖保健品/高蛋白边界。----
MSD_NUTRITION = [
    ("nutrition-obesity", "/home/hormonal-and-metabolic-disorders/obesity-and-the-metabolic-syndrome/obesity", "肥胖", "疾病与健康知识"),
    ("nutrition-metabolic-syndrome", "/home/hormonal-and-metabolic-disorders/obesity-and-the-metabolic-syndrome/metabolic-syndrome", "代谢综合征", "疾病与健康知识"),
    ("nutrition-overview", "/home/disorders-of-nutrition/overview-of-nutrition/overview-of-nutrition", "营养概述", "疾病与健康知识"),
    ("nutrition-calories", "/home/disorders-of-nutrition/overview-of-nutrition/calories", "卡路里", "疾病与健康知识"),
    ("nutrition-macros", "/home/disorders-of-nutrition/overview-of-nutrition/carbohydrates-proteins-and-fats", "碳水化合物、蛋白质和脂肪", "疾病与健康知识"),
    ("nutrition-requirements", "/home/disorders-of-nutrition/overview-of-nutrition/nutritional-requirements", "营养需求", "疾病与健康知识"),
    ("nutrition-vitamins-minerals", "/home/disorders-of-nutrition/overview-of-nutrition/vitamins-and-minerals", "维生素和矿物质", "疾病与健康知识"),
    ("nutrition-exercise-overview", "/home/special-subjects/exercise-and-fitness/overview-of-exercise", "锻炼概述", "疾病与健康知识"),
    ("nutrition-exercise-starting", "/home/special-subjects/exercise-and-fitness/starting-an-exercise-program", "开始锻炼计划", "疾病与健康知识"),
    ("nutrition-exercise-safety", "/home/special-subjects/exercise-and-fitness/exercising-safely", "安全进行锻炼", "疾病与健康知识"),
    ("nutrition-chronic-kidney-disease", "/home/kidney-disorders/kidney-failure/chronic-kidney-disease", "慢性肾病", "疾病与健康知识"),
    ("nutrition-supplements-overview", "/home/special-subjects/dietary-supplements-and-vitamins/overview-of-dietary-supplements", "膳食补充剂概述", "用药安全与特殊人群"),
]

# ---- 运动睡眠与生活习惯 + 常见症状疾病（2026-10-21 第八轮 P1；URL 逐个 curl 验证 200）----
# 覆盖任务清单 P1 两组问题。已知无独立词条、诚实记缺口：咖啡因/咖啡量、
# 长时间用电脑护眼（MSD 大众版无对应中文词条，记待补充来源）。
MSD_P1 = [
    # 运动、睡眠与生活习惯
    ("p1-sleep-overview", "/home/brain-spinal-cord-and-nerve-disorders/sleep-disorders/overview-of-sleep", "睡眠概述", "症状与就医指引"),
    ("p1-snoring", "/home/brain-spinal-cord-and-nerve-disorders/sleep-disorders/snoring", "打鼾", "症状与就医指引"),
    ("p1-circadian-rhythm", "/home/brain-spinal-cord-and-nerve-disorders/sleep-disorders/circadian-rhythm-sleep-disorders", "昼夜节律失调性睡眠障碍", "症状与就医指引"),
    ("p1-fatigue", "/home/special-subjects/nonspecific-symptoms/fatigue", "疲劳", "症状与就医指引"),
    ("p1-alcohol-use", "/home/special-subjects/illicit-drugs-and-intoxicants/alcohol-use", "酒精使用", "疾病与健康知识"),
    ("p1-smoking", "/home/special-subjects/tobacco-use/smoking-and-other-tobacco-use", "吸烟和其他烟草使用", "疾病与健康知识"),
    ("p1-smoking-cessation", "/home/special-subjects/tobacco-use/smoking-cessation", "戒烟", "疾病与健康知识"),
    ("p1-musculoskeletal-pain", "/home/bone-joint-and-muscle-disorders/symptoms-of-musculoskeletal-disorders/musculoskeletal-pain", "肌肉骨骼疼痛", "症状与就医指引"),
    ("p1-choosing-exercise", "/home/special-subjects/exercise-and-fitness/choosing-the-right-exercise", "选择合适的运动", "疾病与健康知识"),
    # 补收：覆盖「久坐/缺乏运动的危害」表述（实测「久坐人群」问法下锻炼概述未进 top-10，
    # 该词条从缺乏运动危害角度叙述，URL 验证 200）
    ("p1-exercise-benefits", "/home/special-subjects/exercise-and-fitness/benefits-of-exercise", "锻炼的益处", "疾病与健康知识"),    # 常见症状与疾病科普
    ("p1-indigestion", "/home/digestive-disorders/symptoms-of-digestive-disorders/indigestion", "消化不良", "症状与就医指引"),
    ("p1-gas", "/home/digestive-disorders/symptoms-of-digestive-disorders/gas", "胃肠气胀（腹胀）", "症状与就医指引"),
    ("p1-influenza", "/home/infections/respiratory-virus-infections/influenza-flu", "流行性感冒（流感）", "疾病与健康知识"),
    ("p1-antibiotics-overview", "/home/infections/antibiotics/overview-of-antibiotics", "抗生素概述", "用药安全与特殊人群"),
    ("p1-steatotic-liver", "/home/liver-and-gallbladder-disorders/manifestations-of-liver-disease/steatotic-liver-disease", "脂肪性肝病", "疾病与健康知识"),
    ("p1-gerd", "/home/digestive-disorders/esophageal-and-swallowing-disorders/gastroesophageal-reflux-disease-gerd", "胃食管反流病", "疾病与健康知识"),
    ("p1-diabetes-overview", "/home/hormonal-and-metabolic-disorders/diabetes-mellitus-and-low-blood-sugar-hypoglycemia/overview-of-diabetes-mellitus", "糖尿病概述", "疾病与健康知识"),
    ("p1-itching", "/home/skin-disorders/itching-and-dermatitis/itching", "瘙痒", "症状与就医指引"),
]

# ---- 就医指引与急救 + 用药安全与特殊人群（2026-10-21 第九轮；URL 逐个 curl 验证 200）----
# 覆盖任务清单 P0 就医指引 12 题与 P2 用药安全/特殊人群 17 题。检查指标 17 题（P1）
# 由既有化验/影像词条覆盖，本轮不新增文档。
# 注：紧急风险问题主要由安全规则层拦截（emergency_handled），急救词条是兜底证据。
MSD_P2 = [
    ("p0-first-aid-steps", "/home/injuries-and-poisoning/first-aid/steps-in-emergency-first-aid", "急救的步骤", "症状与就医指引"),
    ("p0-internal-bleeding", "/home/injuries-and-poisoning/first-aid/internal-bleeding", "内出血", "症状与就医指引"),
    ("p0-anaphylaxis", "/home/immune-disorders/allergic-reactions-and-other-hypersensitivity-disorders/anaphylactic-reactions", "速发严重过敏反应", "症状与就医指引"),
    ("p2-seizures", "/home/brain-spinal-cord-and-nerve-disorders/seizure-disorders/seizure-disorders", "癫痫（抽搐）", "症状与就医指引"),
    ("p2-med-safety-overview", "/home/drugs/overview-of-medications/overview-of-medication-safety-and-effectiveness", "药物安全性和有效性概述", "用药安全与特殊人群"),
    ("p2-children-fever", "/home/children-s-health-issues/symptoms-in-infants-and-children/fever-in-infants-and-children", "婴儿和儿童发热", "用药安全与特殊人群"),
    ("p2-falls-older", "/home/older-people-s-health-issues/falls-in-older-adults/falls-in-older-adults", "老年人跌倒", "用药安全与特殊人群"),
    ("p2-menopause", "/home/women-s-health-issues/menopause/menopause", "更年期", "用药安全与特殊人群"),
]

# ---- 72 题覆盖测试暴露缺口的补充（2026-10-21 第十轮；URL 逐个 curl 验证 200）----
# 依据 docs/覆盖分析报告-72题-20261021.md 阶段 4 清单：免疫接种(K05)、尿常规解读
# (E05)、手足口病(H05)、多重用药/老年评估(C06)、孕期不适(G05)、癌症筛查(K04)。
# 儿童用药剂量(G06) MSD 无词条且官方指南原文未能定位获取，记待补充来源（未完成）。
MSD_EXTRA = [
    ("ext-immunization-overview", "/home/infections/immunization/overview-of-immunization", "免疫接种概述", "疾病与健康知识"),
    ("ext-influenza-vaccine", "/home/infections/immunization/influenza-vaccine", "流感疫苗", "疾病与健康知识"),
    ("ext-blood-in-urine", "/home/kidney-disorders/symptoms-of-kidney-disorders/blood-in-urine", "血尿", "检查指标解读"),
    ("ext-asymptomatic-proteinuria", "/home/kidney-disorders/kidney-filtering-disorders/asymptomatic-proteinuria-and-hematuria", "无症状性蛋白尿和血尿", "检查指标解读"),
    ("ext-hand-foot-mouth", "/home/infections/enteroviruses/hand-foot-and-mouth-disease", "手足口病", "疾病与健康知识"),
    ("ext-aging-and-medications", "/home/older-people-s-health-issues/aging-and-medications/aging-and-medications", "衰老和药物", "用药安全与特殊人群"),
    ("ext-therapeutic-objectives-older", "/home/older-people-s-health-issues/aging-and-quality-of-life/therapeutic-objectives-in-older-adults", "老年人的治疗目标", "用药安全与特殊人群"),
    ("ext-pregnancy-self-care", "/home/women-s-health-issues/normal-pregnancy/self-care-during-pregnancy", "孕期自我保健", "用药安全与特殊人群"),
    ("ext-pregnancy-physical-changes", "/home/women-s-health-issues/normal-pregnancy/physical-changes-during-pregnancy", "孕期母体变化", "用药安全与特殊人群"),
    ("ext-delirium", "/home/brain-spinal-cord-and-nerve-disorders/delirium-and-dementia/delirium", "谵妄（急性意识模糊）", "疾病与健康知识"),
    ("ext-postpartum-care", "/home/women-s-health-issues/postpartum-care/overview-of-postpartum-care", "产后护理概述", "用药安全与特殊人群"),
]

# WHO 中文实况报道（权威国际来源；全 URL 直收，fetch.py 用通用正文抽取器）
WHO_EXTRA = [
    ("who-cancer-fact-sheet", "https://www.who.int/zh/news-room/fact-sheets/detail/cancer",
     "癌症（WHO 实况报道）", "疾病与健康知识"),
]

ALL_MSD = (MSD_DISEASES + MSD_SYMPTOMS + MSD_TESTS + MSD_MED_SAFETY
           + MSD_COMMON_HEALTH + MSD_NUTRITION + MSD_P1 + MSD_P2 + MSD_EXTRA)
