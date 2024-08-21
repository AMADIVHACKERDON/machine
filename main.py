from pydantic import BaseModel
from fastapi import FastAPI, Response,status,HTTPException
from random import randrange
app = FastAPI()

class Post(BaseModel):
    title: str
    content: str
    published: bool

mypost = [{"title":"title of post 1", "content":"content of post 1","id": 2}, {
    'title':'favourite foods', 'content':'I like Pizza', 'id':1}]

def find_post(id):
    for p in mypost:
        if p['id'] == id:
            return p
def find_index_post(id):
    for i,p in enumerate(mypost):
        if p['id'] == id:
            return i

@app.get('/')
async def root():
    return {"message":'Hello world'}

@app.get("/posts")
def get_post():
    return {"data": mypost}

@app.post("/posts")
def create_post(new_post: Post):
    post_dict = new_post.dict()
    post_dict['id'] = randrange(0,1000)
    mypost.append(post_dict)
    return {"data":post_dict}
@app.get("/posts/{id}")
def getpost(id:int, response:Response):

    post = find_post(id)
    if not post:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'message':f'post with id: {id} was not found '}
    return {"post-detail": post}

@app.delete("/posts/{id}",status_code=status.HTTP_204_NO_CONTENT)
def delete_post(id:int):
    index = find_index_post(id)

    if index == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'post with id:{id} does not exist')

    my_posts.pop(index)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.put("/posts/{id}")
def update_post(id:int, post:Post):
    index = find_index_post(id)

    if index == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'post with id:{id} does not exist')

    post_dict = post.dict()
    post_dict['id'] = id
    mypost[index] =post_dict
    return {'data':post_dict}