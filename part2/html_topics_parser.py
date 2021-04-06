from html.parser import HTMLParser

class TopicsParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.last_tag = None
        self.topics_list=[]
        self.dictionary={}

    def handle_starttag(self, tag, attrs):
        self.last_tag=tag

    def handle_endtag(self, tag):
        self.last_tag='top'
        self.topics_list.append(self.dictionary.copy())
        self.dictionary.clear()
        
    def handle_data(self, data):
        if self.last_tag!='top':
            if self.last_tag!='title':
                _index=data.find(':')
                if _index!=-1:
                    data=data[_index+1:].strip()
            
            self.dictionary[self.last_tag]=data
    
    def get_data(self,file_path):
        file=open(file_path,'r')
        self.feed(file.read().lower())
        return self.topics_list.copy()

'''
#Uncomment to test
parser = TopicsParser()
topics=parser.get_data('topics.txt')
print(parser.topics_list)
'''
