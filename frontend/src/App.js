import { useState } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown'

function App() {
  const [message, setMessage] = useState("");
  const [chatList, setChatList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false); // 上传状态
  // 后端地址统一从配置读取（需求 3.4.5：不分别硬编码不同主机地址）；
  // 未配置时回退到本地默认值，行为与改动前一致。
  const API_BASE = process.env.REACT_APP_API_BASE || `http://127.0.0.1:8000`;
  // 会话标识：页面加载时生成一次，同一页面内的连续追问共享上下文（N02 多轮）。
  const [sessionId] = useState(() =>
    `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
  );
  // 发送消息 -> 调用后端 /agent/chat（主 Agent 编排链路，同步返回完整回答）
  const sendChat = () => {
    if (!message.trim() || loading) return; // 防止连点
    const question = message.trim();
    setChatList(prev => [...prev, { role: "user", content: question }]);
    setMessage("");
    setLoading(true);

    // 追加一个空的AI回复占位
    setChatList(prev => [...prev, { role: "ai", content: "" }]);

    fetch(`${API_BASE}/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, question }),
    })
      .then(res => res.json())
      .then(data => {
        // 后端已把正文、风险提示、来源、免责声明组装进 answer；非 200 时接口返回
        // {detail}（固定话术，不含内部信息），故优先取 answer，其次 detail。
        const content = data.answer || data.detail || "服务暂时不可用，请稍后重试。";
        setChatList(prev => {
          const next = [...prev];
          next[next.length - 1] = { role: "ai", content };
          return next;
        });
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setChatList(prev => {
          const next = [...prev];
          next[next.length - 1] = { role: "ai", content: "网络请求失败，请检查后端服务。" };
          return next;
        });
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
      const response = await fetch(`${API_BASE}/upload`, {
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