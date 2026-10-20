import { useState, useEffect, useRef } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown'

// 后端 /agent/chat 的 outcome（TerminationReason）→ 前端展示标签。
// 用户可见话术已由后端组装进 answer，这里只做状态指示，不再复述文案。
const OUTCOME_META = {
  emergency_handled: { text: '紧急提示', cls: 'badge-emergency' },
  insufficient_evidence: { text: '证据不足', cls: 'badge-muted' },
  unclassified: { text: '无法处理', cls: 'badge-muted' },
  context_missing: { text: '请补充说明', cls: 'badge-muted' },
  retrieval_failed: { text: '检索异常', cls: 'badge-muted' },
  empty_knowledge_base: { text: '知识库无内容', cls: 'badge-muted' },
  budget_exhausted: { text: '超时', cls: 'badge-muted' },
  generation_failed: { text: '生成失败', cls: 'badge-muted' },
};
// 与后端 rag/service.py MAX_QUESTION_LEN 对齐（N06 校验口径，测试 T-07）
const MAX_QUESTION_LEN = 500;

function App() {
  const [message, setMessage] = useState("");
  const [chatList, setChatList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false); // 上传状态
  const chatBoxRef = useRef(null); // 对话区自动滚动到底
  // 后端地址统一从配置读取（需求 3.4.5：不分别硬编码不同主机地址）；
  // 未配置时回退到本地默认值，行为与改动前一致。
  const API_BASE = process.env.REACT_APP_API_BASE || `http://127.0.0.1:8000`;
  // 会话标识：localStorage 持久化（设计 C-6 确认项）。刷新/重开页面沿用同一会话，
  // 多轮上下文（N02）不因页面关闭丢失；后端服务重启后会话存储清空，同一 session_id
  // 按新会话处理并由后端给出无历史说明（现有 N02/降级逻辑，无需前端特判）。
  const SESSION_KEY = "medical-assistant-session-id";
  const [sessionId] = useState(() => {
    let sid = null;
    try {
      sid = window.localStorage.getItem(SESSION_KEY);
      if (!sid) {
        sid = `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
        window.localStorage.setItem(SESSION_KEY, sid);
      }
    } catch {
      // localStorage 不可用（隐私模式等）时退化为每次加载新会话，行为同改动前
      sid = `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }
    return sid;
  });

  // 新消息或加载状态变化时，滚动到对话区底部
  useEffect(() => {
    const el = chatBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatList, loading]);

  // 发送消息 -> 调用后端 /agent/chat（主 Agent 编排链路，同步返回完整回答）
  const sendChat = () => {
    if (!message.trim() || loading) return; // 防止连点
    const question = message.trim();
    if (question.length > MAX_QUESTION_LEN) {
      setChatList(prev => [...prev,
        { role: "ai", content: `问题过长（${MAX_QUESTION_LEN} 字以内），请精简后再发。`, outcome: "unclassified" },
      ]);
      return;
    }
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
          next[next.length - 1] = { role: "ai", content, outcome: data.outcome || null };
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
      // /upload 解析/入库失败时后端仍返回 HTTP 200，失败信息在 message 里
      // （已记录的口径缝隙）：前端按措辞识别，避免把失败误报成成功。
      const text = data.message || "上传成功";
      const failed = !response.ok || text.includes("处理失败");
      setChatList(prev => [...prev, {
        role: "ai",
        content: text,
        outcome: failed ? "retrieval_failed" : null,
      }]);
    } catch (error) {
      console.error("上传失败:", error);
      setChatList(prev => [...prev, {
        role: "ai", content: "上传失败，请检查后端服务", outcome: "retrieval_failed",
      }]);
    } finally {
      setUploading(false);
      e.target.value = ''; // 清空 input 的值，允许再次上传同一个文件
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <h1>医疗科普问答助手</h1>
          <p>基于知识库证据回答 · 紧急情况请立即拨打 <strong>120</strong></p>
        </div>
      </header>

      <main className="chat-container">
        <div className="chat-box" ref={chatBoxRef}>
          {chatList.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">🩺</div>
              <p>您好，我是医疗科普问答助手</p>
              <p className="welcome-sub">请描述您的健康问题，我会依据知识库资料回答；
                也可以问我药品用法、检查指标、疾病科普等内容。</p>
            </div>
          )}
          {chatList.map((item, index) => {
            const meta = item.outcome ? OUTCOME_META[item.outcome] : null;
            return (
              <div key={index} className={`msg-row ${item.role}`}>
                <div className={`bubble ${item.role}`}>
                  {meta && <span className={`outcome-badge ${meta.cls}`}>{meta.text}</span>}
                  {item.role === 'ai' ? (
                    <div className="markdown-body">
                      <ReactMarkdown>{item.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p>{item.content}</p>
                  )}
                </div>
              </div>
            );
          })}
          {loading && (
            <div className="msg-row ai">
              <div className="bubble ai loading">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            </div>
          )}
        </div>

        <div className="input-area">
          <label className="upload-btn">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            <span>{uploading ? '上传中' : '上传PDF'}</span>
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
            placeholder={`请输入您的问题...（${MAX_QUESTION_LEN} 字以内）`}
            maxLength={MAX_QUESTION_LEN}
            onKeyDown={e => e.key === 'Enter' && sendChat()}
          />
          <button onClick={sendChat} disabled={loading}>发送</button>
        </div>
      </main>
    </div>
  );
}

export default App;
