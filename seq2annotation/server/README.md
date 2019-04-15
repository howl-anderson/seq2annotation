## HTTP server
### run service
```bash
python -m seq2annotation.server.http /path/to/saved_model
```

默认启动在 主机： `localhost` 端口：`5000`
### input format
example:
in HTTP format
```text
GET /parse?q=播放周杰伦的叶惠美 HTTP/1.1
Host: howl-MS-7A67:5000
```
### output
example:
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
