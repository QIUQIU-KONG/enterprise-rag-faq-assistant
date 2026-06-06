"""
Domain definitions and subtopic trees for the knowledge base.
"""
from enum import Enum


class Domain(str, Enum):
    TRAVEL_TIPS = "travel_tips"
    MALAYSIA_VISA = "malaysia_visa"
    PROJECT_APPLICATIONS = "project_applications"


# User-facing labels (single source of truth — do NOT duplicate in other files)
DOMAIN_LABELS: dict[str, str] = {
    Domain.TRAVEL_TIPS: "出差注意事项（物品准备、安全、文化禁忌等）",
    Domain.MALAYSIA_VISA: "马来西亚商务签证（材料、流程、时间周期）",
    Domain.PROJECT_APPLICATIONS: "项目申报材料（政府文件、表单填写、提交流程）",
}

# Short labels for UI buttons, options, and compact displays
DOMAIN_SHORT_LABELS: dict[str, str] = {
    Domain.TRAVEL_TIPS: "出差注意事项",
    Domain.MALAYSIA_VISA: "马来西亚签证办理",
    Domain.PROJECT_APPLICATIONS: "项目申报材料",
}

# Subtopic trees for each domain
DOMAIN_SUBTOPICS: dict[str, list[dict]] = {
    Domain.TRAVEL_TIPS: [
        {
            "id": "packing",
            "title": "物品准备清单",
            "description": "出差马来西亚需要携带的证件、衣物、电子设备等物品",
            "topics": [
                "必备证件（护照、签证、身份证、派遣函）",
                "衣物建议（当地气候、商务着装要求）",
                "电子设备（转换插头、充电器、移动电源规定）",
                "药品与健康（常用药品、防蚊措施、疫苗接种）",
                "货币与支付（马币兑换、信用卡使用、小费习惯）",
            ],
        },
        {
            "id": "safety",
            "title": "安全注意事项",
            "description": "在马来西亚出差期间的人身、财物安全须知",
            "topics": [
                "人身安全（夜间出行、高风险区域、紧急联系方式）",
                "财物安全（贵重物品保管、酒店保险箱使用）",
                "交通安全（当地驾驶规则、网约车使用建议）",
                "食品安全（饮用水、街头食品、餐厅选择）",
                "自然灾害（雨季防范、蚊虫防护）",
            ],
        },
        {
            "id": "culture_taboos",
            "title": "文化禁忌与商务礼仪",
            "description": "马来西亚当地的宗教文化禁忌和商务社交礼仪",
            "topics": [
                "宗教禁忌（伊斯兰教礼仪、清真寺参观注意事项）",
                "社交礼仪（握手方式、名片交换、称呼习惯）",
                "饮食禁忌（清真食品规定、饮酒限制、用餐礼仪）",
                "着装要求（商务场合、日常出行、宗教场所）",
                "节日与假期（主要节日、斋月期间注意事项）",
            ],
        },
        {
            "id": "transport",
            "title": "交通出行指南",
            "description": "马来西亚当地交通方式和出行建议",
            "topics": [
                "机场交通（KLIA到市区、机场大巴和快线）",
                "市内出行（Grab打车、地铁LRT、出租车注意事项）",
                "跨城交通（国内航班、长途巴士、火车）",
                "租车自驾（驾照要求、保险、路况说明）",
            ],
        },
    ],
    Domain.MALAYSIA_VISA: [
        {
            "id": "materials",
            "title": "签证办理材料",
            "description": "马来西亚商务签证申请所需全部材料清单及要求",
            "topics": [
                "个人基本材料（护照、照片、身份证复印件、户口本）",
                "公司证明材料（在职证明、营业执照副本、派遣函）",
                "邀请方材料（马来西亚公司邀请函、注册证明）",
                "财务证明材料（银行流水、存款证明、工资单）",
                "行程材料（机票预订单、酒店预订单、行程安排表）",
            ],
        },
        {
            "id": "process",
            "title": "签证办理流程",
            "description": "马来西亚商务签证从申请到领取的完整流程",
            "topics": [
                "线上申请（eVISA系统操作、申请表填写指南）",
                "材料递交（使馆/签证中心地址、预约方式、递交时间）",
                "生物信息采集（指纹录入、照片采集要求）",
                "缴费方式（签证费用标准、支付方式）",
                "领取护照（领取方式、代领要求、邮寄选项）",
            ],
        },
        {
            "id": "timeline",
            "title": "办理时间周期",
            "description": "马来西亚商务签证各环节的时间节点",
            "topics": [
                "常规办理周期（标准审理时间、建议提前多久申请）",
                "加急办理（加急条件、加急费用、最快出签时间）",
                "签证有效期（单次/多次入境有效期、停留天数限制）",
                "高峰期提醒（节假日前后、寒暑假办理建议）",
            ],
        },
        {
            "id": "faq",
            "title": "签证常见问题",
            "description": "马来西亚商务签证办理过程中的高频问题",
            "topics": [
                "拒签常见原因及预防（材料不全、信息不一致、资金不足）",
                "签证延期（延期条件、办理方式、费用）",
                "签证转换（旅游签转商务签的可行性）",
                "eVISA与贴纸签区别（适用场景、优劣势对比）",
                "被拒后重新申请（等待时间、材料补充建议）",
            ],
        },
    ],
    Domain.PROJECT_APPLICATIONS: [
        {
            "id": "government_docs",
            "title": "政府红头文件要求",
            "description": "各级政府项目申报的官方文件要求和解读",
            "topics": [
                "申报指南解读（如何阅读和理解政府申报通知）",
                "资质要求（企业资质等级、注册年限、行业限制）",
                "必备材料清单（营业执照、财务报表、项目计划书）",
                "格式规范（文件装订要求、电子版格式、盖章要求）",
                "政策支持方向（重点支持领域、优先立项条件）",
            ],
        },
        {
            "id": "form_fields",
            "title": "政务网表单填写指南",
            "description": "各类政务网在线申报系统的表单字段说明",
            "topics": [
                "基本信息填报（企业统一信用代码、法人信息、联系人）",
                "项目详情填报（项目名称规范、技术路线描述、创新点提炼）",
                "预算填报（经费科目分类、自筹资金比例、预算说明要求）",
                "附件上传规范（文件格式、大小限制、命名规则）",
                "常见填写错误（数据单位混淆、日期格式、金额大小写）",
            ],
        },
        {
            "id": "submission",
            "title": "提交流程与时间节点",
            "description": "项目申报的完整提交流程和关键时间节点",
            "topics": [
                "申报时间线（通知发布、材料准备、网上填报、纸质提交）",
                "审核流程（形式审查→专家评审→公示立项→签订合同）",
                "常见退回原因及处理（补充材料、修改说明、重新提交）",
                "多部门联合申报（牵头单位要求、合作协议模板）",
                "项目变更申请（延期申请、预算调整、人员变更流程）",
            ],
        },
        {
            "id": "common_mistakes",
            "title": "申报常见错误与避坑指南",
            "description": "项目申报过程中最常犯的错误及如何避免",
            "topics": [
                "材料准备阶段（漏盖章、数据不一致、证件过期）",
                "系统填报阶段（超时自动退出、未保存草稿、浏览器兼容）",
                "审核反馈后（未在时限内补充、补充材料仍不合格）",
                "预算编制（科目不符、比例超标、依据不充分）",
            ],
        },
    ],
}
