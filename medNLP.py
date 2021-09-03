# %%
import xml.etree.ElementTree as ET


tree = ET.parse('MedTxt-CR-JA-training.xml')
# %%
root = tree.getroot()

# %%
root[1][0].attrib
# %%
print(root)
# %%
for article in root.iter('article'):
    print(article.attrib)
    print(article.text)
# %%
root[1][0].findall('article')
# %%
for a in root.findall('articles'):
   paragraph= a.findall('article')
   for a in paragraph:
       print(a.text)

# %%
for article in root.iter('articles'):
    print(article)
# %%
text = ET.tostring(root[1][0], encoding='unicode')
# %%
tree = ET.parse(text)
# %%
tree
# %%
import codecs
from bs4 import BeautifulSoup
with codecs.open("MedTxt-CR-JA-training.xml", "r", "utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")
print(soup.prettify())
# %%
print(soup.find("article"))
# %%
print(soup.find("article").prettify())
# %%
docs = []

for elem in soup.find_all("article"):
    texts = []
    for child in elem.find_all('NavigableString'):
        print(child.string)
        print(child.name)
        if child.name == None:
            print(child.extract())
    """
        if child.name != None:
            label = 'I'
            type_id = child.name
        else:
            label = 'O'
        for w in child.text.split(" "):
            if len(w) > 0:
                texts.append((w, label))
    docs.append(texts)
    """
docs
# %%
soup.find_all('article')
# %%
entities = []

for elem in soup.find_all("articles"):
    texts = []
    pos1 = 0
    pos2 = 0
    for child in elem.find("article"):
        print(child.string)
        
        print(type(child))


        pos += len(child.string)
        if child.name != None:
            {'name':child.string, 'span':[pos1, pos2], type:}
        print(pos)
    """
        if type(child) == Tag:
            if child.name == "namedentityintext":
                label = 'N'
            else:
                label = 'C'
            for w in child.text.split(" "):
                if len(w) > 0:
                    texts.append((w, label))
    docs.append(texts)
    """
# %%
