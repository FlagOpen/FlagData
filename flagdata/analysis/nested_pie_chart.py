import plotly.express as px

# Your provided data
data = {
    "理解能力": {
        "Total_number": 40,
        "Secondary_classification": {
            "信息分析": 10,
            "信息提取": 10,
            "信息概括": 10,
            "跨语言理解": 5,
            "文化理解": 4,
            "判别评价": 1
        }
    },
    "知识运用": {
        "Total_number": 20,
        "Secondary_classification": {
            "知识问答": 10,
            "常识问答": 5,
            "事实问答": 5
        }
    },
    "推理能力": {
        "Total_number": 10,
        "Secondary_classification": {
            "知识推理": 5,
            "符号推理": 5
        }
    },
    "特殊生成": {
        "Total_number": 7,
        "Secondary_classification": {
            "条件生成": 4,
            "代码生成": 2,
            "创意生成": 1
        }
    },
    "解释能力": {
        "Total_number": 5,
        "Secondary_classification": {
            "心理疏导": 3,
            "语言解析": 2
        }
    },
    "价值观": {
        "Total_number": 5,
        "Secondary_classification": {
            "歧视偏见": 2,
            "安全": 2,
            "伦理道德": 1
        }
    },
    "幻觉": {
        "Total_number": 3,
        "Secondary_classification": {}
    },
    "通用综合能力": {
        "Total_number": 2,
        "Secondary_classification": {}
    },
    "领域专家能力": {
        "Total_number": 1,
        "Secondary_classification": {}
    }
}


# Function to convert nested data to a list of dictionaries suitable for Sunburst Chart
def convert_to_sunburst(data, name='', parent=None):
    result = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and 'Secondary_classification' in value:
                result += convert_to_sunburst(value['Secondary_classification'], name + key + '/', name + key)
            else:
                result.append({'id': name + key, 'parent': parent, 'value': value})
    return result


sunburst_data = convert_to_sunburst(data)

# Create Sunburst Chart using Plotly Express
fig = px.sunburst(sunburst_data, path=['parent', 'id'], values='value')
fig.show()
