## HTTP server
### run service
```bash
python -m seq2annotation.server.http /path/to/saved_model
```

默认启动在 主机： `localhost` 端口：`5000`

## 输入/输出格式 ##
HTTP 服务接受 GET 和 POST 方式的请求。

GET 方式接受单个请求（字符串），返回单个响应 JSON 结构（称之为 Response Block)。适合正式用户请求和少量测试用。

POST 方式接受多个请求，请求以 JSON 列表的形式呈现，每个列表元素都是字符串。返回结构也是 JSON 列表，每个列表元素都是 Response Block。

### GET 方式 

请求 `http://<IP>:<PORT>/parse?q=<USER_TEXT>`

其中：
<IP> 是服务器地址
<PORT> 是服务的端口
<USER_TEXT> 是用户请求的文本，比如 `播放周杰伦的叶惠美`

响应的格式的content-type 是 JSON。内容是：

```json
{
    "ents": [
        "人名",
        "歌曲名"
    ],
    "spans": [
        {
            "end": 5,
            "start": 2,
            "type": "人名"
        },
        {
            "end": 9,
            "start": 6,
            "type": "歌曲名"
        }
    ],
    "text": "播放周杰伦的叶惠美"
}
```

### POST 方式 ###

请求地址 `http://<IP>:<PORT>/parse`

请求 body 格式为 JSON，内容如下：

```json
[
  "播放周杰伦的叶惠美",
  "...",
]
```

响应的格式的content-type 是 JSON。内容是：

```json
[
  {
      "ents": [
          "人名",
          "歌曲名"
      ],
      "spans": [
          {
              "end": 5,
              "start": 2,
              "type": "人名"
          },
          {
              "end": 9,
              "start": 6,
              "type": "歌曲名"
          }
      ],
      "text": "播放周杰伦的叶惠美"
  },
  ...
]
