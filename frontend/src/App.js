import { useState, useEffect, useRef, useCallback } from 'react';
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

// ---- 本地多会话存储（侧边栏）：会话 id 即后端 session_id，后端无需改动 ----
const CONV_KEY = 'medical-assistant-conversations';
const ACTIVE_KEY = 'medical-assistant-active-conv';

function loadConversations() {
  try {
    const raw = window.localStorage.getItem(CONV_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch { return []; }
}

function newConversation() {
  const id = `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return { id, title: '新的对话', createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
}

// 欢迎页快捷提问（对应知识库已覆盖的高频主题）
const SUGGESTIONS = [
  '高血压患者在生活方式上要注意什么？',
  '晚上睡不着怎么办？',
  '盐酸二甲双胍片的用法用量是什么？',
  '痛经怎么办，如何缓解？',
  '普通感冒怎么护理好得快？',
  '血常规检查主要看哪些指标？',
];

// 风险提示文本是 answer 的前缀：横幅已单独渲染时，从正文剥掉避免重复展示。
function stripRiskNotice(content, riskNotice) {
  if (riskNotice && content && content.startsWith(riskNotice)) {
    return content.slice(riskNotice.length).trimStart();
  }
  return content;
}

function App() {
  const [conversations, setConversations] = useState(() => {
    const list = loadConversations();
    return list.length > 0 ? list : [newConversation()];
  });
  const [activeId, setActiveId] = useState(() => {
    try { return window.localStorage.getItem(ACTIVE_KEY); } catch { return null; }
  });
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const chatBoxRef = useRef(null);
  const inputRef = useRef(null);
  // 窄屏（与 App.css 断点一致）用短占位提示，避免长提示被截断出半个括号
  const [narrowScreen, setNarrowScreen] = useState(
    () => window.matchMedia('(max-width: 760px)').matches);
  // 流式输出：全量文本放 ref，可见前缀放 state，切换会话时取消
  const streamRef = useRef(null);

  const API_BASE = process.env.REACT_APP_API_BASE || `http://127.0.0.1:8000`;

  // 输入框高度随内容自适应（参考 ChatGPT）：单行起步，最多约 5 行，超出滚动
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }, [message]);

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 760px)');
    const onChange = e => setNarrowScreen(e.matches);
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else mq.addListener(onChange); // 旧浏览器兜底
    return () => {
      if (mq.removeEventListener) mq.removeEventListener('change', onChange);
      else mq.removeListener(onChange);
    };
  }, []);

  // activeId 失效（首次进入或记录了已删除的会话）时回退到列表首条
  useEffect(() => {
    if (!conversations.some(c => c.id === activeId)) {
      setActiveId(conversations[0].id);
    }
  }, [conversations, activeId]);

  const active = conversations.find(c => c.id === activeId) || null;

  useEffect(() => {
    try { window.localStorage.setItem(CONV_KEY, JSON.stringify(conversations)); } catch {}
  }, [conversations]);

  useEffect(() => {
    try {
      if (activeId) window.localStorage.setItem(ACTIVE_KEY, activeId);
    } catch {}
  }, [activeId]);

  // 新消息或加载状态变化时，滚动到对话区底部
  useEffect(() => {
    const el = chatBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [active?.messages, loading]);

  // 卸载/切换会话时停掉进行中的打字动画
  useEffect(() => () => {
    if (streamRef.current) clearInterval(streamRef.current.timer);
  }, [activeId]);

  const patchActive = useCallback((updater) => {
    setConversations(prev => prev.map(c => c.id === activeId ? updater(c) : c));
  }, [activeId]);

  // ---- 流式渲染：把完整回答逐段显示（后端是同步返回，这里做前端渐进呈现） ----
  const streamAnswer = useCallback((full, onDone) => {
    if (streamRef.current) clearInterval(streamRef.current.timer);
    let i = 0;
    const step = Math.max(4, Math.round(full.length / 120)); // ~120 次刷完
    const state = { timer: null, full };
    state.timer = setInterval(() => {
      i += step;
      const visible = full.slice(0, i);
      patchActive(c => {
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (last && last.role === 'ai') msgs[msgs.length - 1] = { ...last, content: visible };
        return { ...c, messages: msgs };
      });
      if (i >= full.length) {
        clearInterval(state.timer);
        streamRef.current = null;
        onDone && onDone();
      }
    }, 24);
    streamRef.current = state;
  }, [patchActive]);

  // 发送消息 -> 调用后端 /agent/chat（主 Agent 编排链路，同步返回完整回答）
  const sendChat = (text) => {
    const question = (text ?? message).trim();
    if (!question || loading) return; // 防止连点
    if (question.length > MAX_QUESTION_LEN) {
      patchActive(c => ({
        ...c,
        messages: [...c.messages,
          { role: "ai", content: `问题过长（${MAX_QUESTION_LEN} 字以内），请精简后再发。`, outcome: "unclassified" }],
      }));
      return;
    }
    setMessage("");
    setLoading(true);

    patchActive(c => ({
      ...c,
      title: c.messages.length === 0 ? question.slice(0, 18) : c.title,
      updatedAt: Date.now(),
      messages: [...c.messages,
        { role: "user", content: question },
        { role: "ai", content: "" }],
    }));

    fetch(`${API_BASE}/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeId, question }),
    })
      .then(res => res.json())
      .then(data => {
        // 后端已把正文、风险提示、来源、免责声明组装进 answer；非 200 时接口返回
        // {detail}（固定话术，不含内部信息），故优先取 answer，其次 detail。
        const content = data.answer || data.detail || "服务暂时不可用，请稍后重试。";
        patchActive(c => {
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (last && last.role === 'ai') {
            msgs[msgs.length - 1] = {
              ...last,
              outcome: data.outcome || null,
              riskNotice: data.risk_notice || null,
            };
          }
          return { ...c, messages: msgs };
        });
        streamAnswer(content, () => setLoading(false));
      })
      .catch(err => {
        console.error(err);
        patchActive(c => {
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (last && last.role === 'ai') {
            msgs[msgs.length - 1] = { ...last, content: "网络请求失败，请检查后端服务。" };
          }
          return { ...c, messages: msgs };
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
      patchActive(c => ({
        ...c,
        messages: [...c.messages, { role: "ai", content: text, outcome: failed ? "retrieval_failed" : null }],
      }));
    } catch (error) {
      console.error("上传失败:", error);
      patchActive(c => ({
        ...c,
        messages: [...c.messages, { role: "ai", content: "上传失败，请检查后端服务", outcome: "retrieval_failed" }],
      }));
    } finally {
      setUploading(false);
      e.target.value = ''; // 清空 input 的值，允许再次上传同一个文件
    }
  };

  const startNewChat = () => {
    const conv = newConversation();
    setConversations(prev => [conv, ...prev]);
    setActiveId(conv.id);
    setSidebarOpen(false);
  };

  const deleteConversation = (id) => {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id);
      return next.length > 0 ? next : [newConversation()];
    });
  };

  const copyMessage = (content) => {
    navigator.clipboard?.writeText(content).catch(() => {});
  };

  const onInputKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  };

  const messages = active?.messages || [];

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <button className="new-chat-btn" onClick={startNewChat}>
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          新建对话
        </button>
        <div className="conv-list">
          {conversations.map(c => (
            <div key={c.id}
                 className={`conv-item ${c.id === activeId ? 'active' : ''}`}
                 onClick={() => { setActiveId(c.id); setSidebarOpen(false); }}>
              <span className="conv-title">{c.title}</span>
              <button className="conv-del" title="删除对话"
                      onClick={(e) => { e.stopPropagation(); deleteConversation(c.id); }}>×</button>
            </div>
          ))}
        </div>
      </aside>
      {sidebarOpen && <div className="sidebar-mask" onClick={() => setSidebarOpen(false)} />}

      <div className="main-pane">
        <header className="app-header">
          <button className="menu-btn" onClick={() => setSidebarOpen(true)} aria-label="会话列表">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
          </button>
          <div className="app-header-inner">
            <h1>医疗科普问答助手</h1>
            <p>基于知识库证据回答 · 紧急情况请立即拨打 <strong>120</strong></p>
          </div>
        </header>

        <main className="chat-container">
          <div className="chat-box" ref={chatBoxRef}>
            {messages.length === 0 && (
              <div className="welcome">
                <div className="welcome-icon">🩺</div>
                <p>您好，我是医疗科普问答助手</p>
                <p className="welcome-sub">请描述您的健康问题，我会依据知识库资料回答；
                  也可以问我药品用法、检查指标、疾病科普等内容。</p>
                <div className="suggestions">
                  {SUGGESTIONS.map(s => (
                    <button key={s} className="suggestion-chip" onClick={() => sendChat(s)}>{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((item, index) => {
              const meta = item.outcome ? OUTCOME_META[item.outcome] : null;
              const isEmergency = item.outcome === 'emergency_handled';
              return (
                <div key={index} className={`msg-row ${item.role}`}>
                  <div className={`bubble ${item.role} ${isEmergency ? 'emergency' : ''}`}>
                    {meta && <span className={`outcome-badge ${meta.cls}`}>{meta.text}</span>}
                    {item.role === 'ai' ? (
                      <>
                        {item.riskNotice && (
                          <div className="risk-banner">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            <span>{item.riskNotice.replace(/^提示：/, '')}</span>
                          </div>
                        )}
                        <div className="markdown-body">
                          <ReactMarkdown>{stripRiskNotice(item.content, item.riskNotice)}</ReactMarkdown>
                        </div>
                        {item.content && (
                          <div className="msg-actions">
                            <button className="copy-btn" title="复制"
                                    onClick={() => copyMessage(item.content)}>
                              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                              复制
                            </button>
                          </div>
                        )}
                      </>
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
                  <span className="thinking-label">思考中</span>
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

            <textarea
              ref={inputRef}
              value={message}
              onChange={e => setMessage(e.target.value)}
              placeholder={narrowScreen
                ? '请输入您的问题...'
                : `请输入您的问题...（Enter 发送，Shift+Enter 换行，${MAX_QUESTION_LEN} 字以内）`}
              maxLength={MAX_QUESTION_LEN}
              rows={1}
              onKeyDown={onInputKeyDown}
            />
            <button onClick={() => sendChat()} disabled={loading}>发送</button>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
