import xml.etree.ElementTree as ET
from collections import namedtuple

utilFunc = namedtuple('utilFunc', 'id, func, a, b')
comp_dict = {
    "greater_or_equal" : lambda x,y : x >= y,
    "less_or_equal" : lambda x,y : x <= y,
    "equal" : lambda x,y : x == y,
    "greater" : lambda x,y : x > y,
    "less" : lambda x,y : x < y,
}

class GSNModel:
    def __init__(self, filename):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        self.utility_funcs = []
        for child in self.root:
            if child.tag == 'GSNSolution':
                solution_id = child.attrib['id']
                comp = child[0].attrib['type']
                comp = child[0].attrib['type']
                leaves = child[0].findall('GSNParameter')
                a = leaves[0].attrib['name']
                b = leaves[1].attrib['name']
                self.utility_funcs.append(utilFunc(solution_id, comp_dict[comp], a, b))

if __name__ == "__main__":
    model = GSNModel('gsn/gsn.xml')