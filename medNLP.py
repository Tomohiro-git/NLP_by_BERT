
# %%
import codecs
from bs4 import BeautifulSoup
with codecs.open("MedTxt-CR-JA-training.xml", "r", "utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")


# %%
def entities_from_xml(file_name):
    import codecs
    from bs4 import BeautifulSoup
    with codecs.open(file_name, "r", "utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    for elem_articles in soup.find_all("articles"):
        entities = []
        articles = []
        for elem in elem_articles.find_all('article'):
            entities_article = []
            text_list = []
            pos1 = 0
            pos2 = 0
            for child in elem:
                text = child.string

                pos2 += len(child.string)
                if child.name != None:
                    entities_article.append({'name':child.string, 'span':[pos1, pos2], 'type':child.name})
                pos1 = pos2
                text_list.append(text)
            articles.append("".join(text_list))
            entities.append(entities_article)
        print(entities)   
    return
    #return {'text':article.string, 'entities':entities}

    
# %%
entities_from_xml('MedTxt-CR-JA-training.xml')

# %%
import codecs
from bs4 import BeautifulSoup
with codecs.open("MedTxt-CR-JA-training.xml", "r", "utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")

elem = soup.find_all('articles')
child = elem[0].find_all('article')
print(child[1])

"""
def entities_from_xml_article(article):
    for child in elem.find_all("article")[0]:
        text = child.string
            try:
                pos2 += len(child.string)
                if child.name != None:
                    entities_article.append({'name':child.string, 'span':[pos1, pos2], 'type':child.name})
                pos1 = pos2
                text_list.append(text)
                print(text)
                print(text_list)
            except TypeError:
                pass
                #print(child)
        try:
            articles.append("".join(text_list))
"""
# %%
