import requests
def sendToWechat(name,content):
    '''
    发送微信消息提醒
    param:  name:消息名称
            content：消息内容
    return: 发送成功为True，失败为False
    '''
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": "9bd43a58df00",
                             "title": "实验提示",
                             "name": str(name),
                             "content": str(content)
                         })
    result = resp.content.decode()
    return 'Success' in result
    
if __name__ == "__main__":  
    sendToWechat('name','content')