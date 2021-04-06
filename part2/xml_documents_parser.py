import xml.etree.ElementTree as ET

class DocumentParser():
    def __init__(self):
        self.tags=['title','headline','byline','dateline','text']
        self.dictionary={}
        
    def parse(self,path):
        self.dictionary.clear()
        try:
            self.root = ET.parse(path).getroot()
        except:
            print("Wrong file path when parsing document!")
        for child in self.root:
            if child.tag in self.tags:
                if child.tag!='text':
                    self.dictionary[child.tag]=child.text
                if child.tag=='text':
                    self.dictionary['text']=''
                    for p in child:
                        self.dictionary['text'] += ' '+p.text
        return self.dictionary


#Uncomment to test
#my_parser=DocumentParser()
#my_dict=my_parser.parse('./rcv1/D_train/19960820/2286newsML.xml')
