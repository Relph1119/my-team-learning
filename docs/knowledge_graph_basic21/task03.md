# Task03 Neo4j图数据库导入数据 {docsify-ignore-all}

## 1 知识梳理

### 1.1 引言
- 图计算：$G=(V,E)$，$V=vertex$(节点)，$E=edge$(边)
- 图数据库：是`NoSQL`（非关系型数据库）的一种，应用图形数据结构的特点（节点、属性和边）存储数据实体和相互之间的关系信息。
- Neo4j数据库：支持ACID、集群、备份和故障转移，通过图形化的界面表示节点和关系，通过 REST 接口或者面向对象的 JAVA API进行访问。

### 1.2 Neo4j简介
- 数据的构成：节点、关系、属性
- 索引：根据索引进行遍历操作，通过创建索引，控制节点或关系，主要用于查找操作和删除操作
- 优势：查询的高性能、设计的灵活性、开发的便捷性、原生图计算引擎、开源的数据库

|优势|描述|
|:---|:---|
| 性能 | 读/写速度快 |
| 数据库设计灵活 | 非结构化数据存储方式 |
| 适应需求变化 | 适合敏捷开发方法 | 
| 易用性 | 用嵌入式、服务器模式、分布式模式等方式使用数据库 |
| 建模便捷 | 使用简单框图设计数据模型 |
| 事务管理 | 支持ACID完整的事务管理特性 |
| 数据增长 | 提供分布式高可用模式，支持大规模的数据增长 |
| 安全性 | 实时备份，恢复数据方便 |
| 表现直观 | 采用图的数据结构直观表现现实世界的应用场景 |

## 2 代码详解
数据源：39健康网。包括15项信息，其中7类实体，约3.7万实体，21万实体关系。  
`build_graph.py`主要是用于数据导入到`Neo4j`数据库中，以下对该代码进行详解。


```python
class MedicalGraph:
    def __init__(self):
        pass
    
    # 读取文件，获得实体，实体关系
    def read_file(self):
        pass
        
    # 创建节点
    def create_node(self, label, nodes):
        pass
    
    # 创建疾病节点的属性
    def create_diseases_nodes(self, disease_info):
        pass
    
    # 创建知识图谱实体
    def create_graphNodes(self):
        pass
    
    # 创建实体关系边
    def create_graphRels(self):
        pass
    
    # 创建实体关系边
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        pass
```


```python
    def __init__(self):
        # 读取文件路径
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 得到数据文件路径
        self.data_path = os.path.join(cur_dir, 'data/disease.csv')
        # 连接Neo4j数据库
        self.graph = Graph("http://localhost:7474", username="neo4j", password="hun1988")
        # 根据文件获得相关的数据，以便能创建节点和关系
        self._diseases, self._symptoms, self._aliases, self._parts, self._departments, \
        self._complications, self._drugs, \
        self._rel_alias, self._rel_symptom, self._rel_part, self._rel_department, \
        self._rel_complication, self._rel_drug, self._diseases_infos \
        = self.read_file()
```


```python
    def read_file(self):
        """
        读取文件，获得实体，实体关系
        :return:
        """
        # cols = ["name", "alias", "part", "age", "infection", "insurance", "department", "checklist", "symptom",
        #         "complication", "treatment", "drug", "period", "rate", "money"]
        # 实体
        diseases = []  # 疾病
        aliases = []  # 别名
        symptoms = []  # 症状
        parts = []  # 部位
        departments = []  # 科室
        complications = []  # 并发症
        drugs = []  # 药品

        # 疾病的属性：age, infection, insurance, checklist, treatment, period, rate, money
        diseases_infos = []
        # 关系
        disease_to_symptom = []  # 疾病与症状关系
        disease_to_alias = []  # 疾病与别名关系
        diseases_to_part = []  # 疾病与部位关系
        disease_to_department = []  # 疾病与科室关系
        disease_to_complication = []  # 疾病与并发症关系
        disease_to_drug = []  # 疾病与药品关系

        all_data = pd.read_csv(self.data_path, encoding='gb18030').loc[:, :].values
        for data in all_data:
            disease_dict = {}  # 疾病信息
            # 疾病
            disease = str(data[0]).replace("...", " ").strip()
            disease_dict["name"] = disease
            # 别名
            line = re.sub("[，、；,.;]", " ", str(data[1])) if str(data[1]) else "未知"
            for alias in line.strip().split():
                aliases.append(alias)
                disease_to_alias.append([disease, alias])
            # 部位
            part_list = str(data[2]).strip().split() if str(data[2]) else "未知"
            for part in part_list:
                parts.append(part)
                diseases_to_part.append([disease, part])
            # 年龄
            age = str(data[3]).strip()
            disease_dict["age"] = age
            # 传染性
            infect = str(data[4]).strip()
            disease_dict["infection"] = infect
            # 医保
            insurance = str(data[5]).strip()
            disease_dict["insurance"] = insurance
            # 科室
            department_list = str(data[6]).strip().split()
            for department in department_list:
                departments.append(department)
                disease_to_department.append([disease, department])
            # 检查项
            check = str(data[7]).strip()
            disease_dict["checklist"] = check
            # 症状
            symptom_list = str(data[8]).replace("...", " ").strip().split()[:-1]
            for symptom in symptom_list:
                symptoms.append(symptom)
                disease_to_symptom.append([disease, symptom])
            # 并发症
            complication_list = str(data[9]).strip().split()[:-1] if str(data[9]) else "未知"
            for complication in complication_list:
                complications.append(complication)
                disease_to_complication.append([disease, complication])
            # 治疗方法
            treat = str(data[10]).strip()[:-4]
            disease_dict["treatment"] = treat
            # 药品
            drug_string = str(data[11]).replace("...", " ").strip()
            for drug in drug_string.split()[:-1]:
                drugs.append(drug)
                disease_to_drug.append([disease, drug])
            # 治愈周期
            period = str(data[12]).strip()
            disease_dict["period"] = period
            # 治愈率
            rate = str(data[13]).strip()
            disease_dict["rate"] = rate
            # 费用
            money = str(data[14]).strip() if str(data[14]) else "未知"
            disease_dict["money"] = money

            diseases_infos.append(disease_dict)

        return set(diseases), set(symptoms), set(aliases), set(parts), set(departments), set(complications), \
               set(drugs), disease_to_alias, disease_to_symptom, diseases_to_part, disease_to_department, \
               disease_to_complication, disease_to_drug, diseases_infos
```


```python
    def create_node(self, label, nodes):
        """
        使用事务完成节点创建
        :param label: 标签
        :param nodes: 节点
        :return:
        """
        tx = self.graph.begin()
        count = 0
        nodes_tmp = []
        for node_name in nodes:
            node = Node(label, name=node_name)
            nodes_tmp.append(node)
            count += 1
            print(count, len(nodes))

        nodes_tmp = Subgraph(nodes_tmp)
        tx.create(nodes_tmp)
        tx.commit()
        return

    def create_diseases_nodes(self, disease_info):
        """
        使用事务创建疾病节点的属性
        :param disease_info: list(Dict)
        :return:
        """
        tx = self.graph.begin()
        nodes = []
        count = 0
        for disease_dict in disease_info:
            node = Node("Disease", name=disease_dict['name'], age=disease_dict['age'],
                        infection=disease_dict['infection'], insurance=disease_dict['insurance'],
                        treatment=disease_dict['treatment'], checklist=disease_dict['checklist'],
                        period=disease_dict['period'], rate=disease_dict['rate'],
                        money=disease_dict['money'])
            nodes.append(node)
            count += 1
            print(count)
        nodes = Subgraph(nodes)
        tx.create(nodes)
        tx.commit()
        return
```


```python
    def create_graphNodes(self):
        """
        创建知识图谱实体
        :return:
        """
        self.create_diseases_nodes(self._diseases_infos)
        self.create_node("Symptom", self._symptoms)
        self.create_node("Alias", self._aliases)
        self.create_node("Part", self._parts)
        self.create_node("Department", self._departments)
        self.create_node("Complication", self._complications)
        self.create_node("Drug", self._drugs)

        return
```


```python
    def create_graphRels(self):
        # 创建各实体的关系
        self.create_relationship("Disease", "Alias", self._rel_alias, "ALIAS_IS", "别名")
        self.create_relationship("Disease", "Symptom", self._rel_symptom, "HAS_SYMPTOM", "症状")
        self.create_relationship("Disease", "Part", self._rel_part, "PART_IS", "发病部位")
        self.create_relationship("Disease", "Department", self._rel_department, "DEPARTMENT_IS", "所属科室")
        self.create_relationship("Disease", "Complication", self._rel_complication, "HAS_COMPLICATION", "并发症")
        self.create_relationship("Disease", "Drug", self._rel_drug, "HAS_DRUG", "药品")

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        """
        创建实体关系边
        :param start_node:
        :param end_node:
        :param edges:
        :param rel_type:
        :param rel_name:
        :return:
        """
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.graph.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return
```


```python
# 调用类的函数，进行数据导入
handler = MedicalGraph()
handler.create_graphNodes()
handler.create_graphRels()
```

数据导入完成之后，访问`Neo4j`的可视化界面，可以看到如下的数据关系图

![](images/task02/02.png)

## 3 思考与讨论

- 可使用事务方式进行节点创建，比迭代创建之后再提交的执行速度要快很多
- 不能使用多线程，由于`py2neo`包里面的提交事务导致不能使用多线程方式进行数据导入，只能采用单一线程处理
