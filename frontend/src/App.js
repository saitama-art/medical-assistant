import { useState } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown'

function App() {
  const [message, setMessage] = useState("");
  const [chatList, setChatList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false); // 上传状态
  const ip = `127.0.0.1`
  // 发送消息 -> 调用后端 /chat/stream 流式接口
  const sendChat = () => {
    if (!message.trim() || loading) return; // 防止连点
    const userMsg = { role: "user", content: message };
    setChatList(prev => [...prev, userMsg]);
    setMessage("");
    setLoading(true);

    // 追加一个空的AI回复占位，后面逐字填充
    setChatList(prev => [...prev, { role: "ai", content: "" }]);

    // 使用 fetch 读取 SSE 流
    fetch(`http://192.168.33.68:8000/chat/stream?question=${encodeURIComponent(message)}`)
      .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        const read = () => {
          reader.read().then(({ done, value }) => {
            if (done) { setLoading(false); return; }
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop(); // 保留不完整行

            for (const line of lines) {
              if (line.startsWith("data:")) {
                const data = line.slice(5).replace(/^\s+/, '');

                if (data === "[DONE]") {
                  setLoading(false);
                  return;
                }

                if (data) {
                  setChatList(prev => {
                    const next = [...prev];
                    // ✅ 绝对纯净的拼接：不加任何额外字符，原样接收！
                    next[next.length - 1] = { role: "ai", content: next[next.length - 1].content + data };
                    return next;
                  });
                }
              }
            }
            read();
          });
        };
        read();
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  // 处理文件上传函数
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`http://${ip}:8000/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      alert(data.message || "上传成功");
    } catch (error) {
      console.error("上传失败:", error);
      alert("上传失败，请检查后端服务");
    } finally {
      setUploading(false);
      e.target.value = ''; // 清空 input 的值，允许再次上传同一个文件
    }
  };

  return (
    <div className="chat-container">
      <h2>医疗科普问答助手</h2>
      <div className="chat-box">
        {chatList.map((item, index) => (
          <div key={index} className={item.role}>
            {item.role === 'ai' ? (
              // ✅ 使用 pre-wrap 保持原始换行，让 ReactMarkdown 去处理符号
              <div className="markdown-body" style={{whiteSpace: 'pre-wrap', wordBreak: 'break-word'}}>
                <ReactMarkdown>{item.content}</ReactMarkdown>
              </div>
            ) : (
               <p>{item.content}</p>
            )}
          </div>
        ))}
        {loading && <div className="ai"><p>正在回答...</p></div>}
      </div>

      <div className="input-area">
        <label className="upload-btn" style={{cursor: 'pointer', color: 'blue', marginRight: '10px'}}>
          {uploading ? '上传中...' : '上传PDF'}
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            style={{display: 'none'}}
          />
        </label>

        <input
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="请输入您的问题..."
          onKeyDown={e => e.key === 'Enter' && sendChat()}
        />
        <button onClick={sendChat} disabled={loading}>发送</button>
      </div>
    </div>
  );
}

export default App;