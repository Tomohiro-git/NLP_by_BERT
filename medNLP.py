
# %%
import codecs
from bs4 import BeautifulSoup
with codecs.open("MedTxt-CR-JA-training.xml", "r", "utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")


# %%
entities = []

for elem in soup.find_all("articles"):
    texts = []
    pos1 = 0
    pos2 = 0
    for child in elem.find("article"):
        pos2 += len(child.string)
        if child.name != None:
            entities.append({'name':child.string, 'span':[pos1, pos2], 'type':child.name})
        pos1 = pos2
# %%
entities
