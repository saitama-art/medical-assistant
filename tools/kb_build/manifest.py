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
]

# 计划内但详情页 id 待解析的常用药（fetch 阶段在平台列表/搜索中解析）
DAYI_WANTED = [
    "阿莫西林", "奥美拉唑", "氯雷他定", "阿托伐他汀", "硝苯地平",
    "缬沙坦", "左氧氟沙星", "蒙脱石散", "对乙酰氨基酚", "头孢克肟",
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
]

ALL_MSD = MSD_DISEASES + MSD_SYMPTOMS + MSD_TESTS + MSD_MED_SAFETY + MSD_COMMON_HEALTH
