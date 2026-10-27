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
  clarification_required: { text: '需要澄清', cls: 'badge-muted' },
  processing_failed: { text: '处理失败', cls: 'badge-muted' },
  safety_rejected: { text: '安全拦截', cls: 'badge-emergency' },
};
// 与后端 rag/service.py MAX_QUESTION_LEN 对齐（N06 校验口径，测试 T-07）
const MAX_QUESTION_LEN = 500;

// ---- API 封装：凭证 Cookie（credentials）+ CSRF 自定义头，见设计方案 §3 ----
const API_BASE = process.env.REACT_APP_API_BASE || `http://127.0.0.1:8000`;
const CSRF_HEADERS = { 'X-Requested-With': 'XMLHttpRequest' };

async function apiFetch(path, { method = 'GET', body } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    credentials: 'include',                     // 跨域携带登录 Cookie（dev 3000→8000）
    headers: { 'Content-Type': 'application/json', ...CSRF_HEADERS },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* 非 JSON 响应按空处理 */ }
  return { status: res.status, data };
}

const errText = (data, fallback) => data?.detail || fallback;

// ---- 服务端会话列表是当前唯一事实源；旧 localStorage 数据仅作一次性导入来源 ----
const LEGACY_CONV_KEY = 'medical-assistant-conversations';
const LEGACY_ACTIVE_KEY = 'medical-assistant-active-conv';

function loadLegacyConversations() {
  try {
    const raw = window.localStorage.getItem(LEGACY_CONV_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter(c => Array.isArray(c.messages) && c.messages.length) : [];
  } catch { return []; }
}

function clearLegacyConversations() {
  try {
    window.localStorage.removeItem(LEGACY_CONV_KEY);
    window.localStorage.removeItem(LEGACY_ACTIVE_KEY);
  } catch { /* 私有模式等场景下静默失败，不影响主流程 */ }
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

// ==================== 登录页 ====================

function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      const { status, data } = await apiFetch('/auth/login', {
        method: 'POST', body: { username: username.trim(), password },
      });
      if (status === 200) { onLogin(data); return; }
      if (status === 401) setError('用户名或密码错误。');
      else if (status === 403) setError(errText(data, '请求被拒绝，请刷新页面重试。'));
      else if (status === 429) setError(errText(data, '尝试过于频繁，请稍后再试。'));
      else setError('登录失败，请稍后重试。');
    } catch {
      setError('网络请求失败，请检查后端服务。');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-shell">
      <form className="login-card" onSubmit={submit}>
        <div className="login-icon">🩺</div>
        <h1>医疗科普问答助手</h1>
        <p className="login-sub">请登录后使用；紧急情况请立即拨打 <strong>120</strong></p>
        <label>
          用户名
          <input value={username} onChange={e => setUsername(e.target.value)}
                 autoComplete="username" required />
        </label>
        <label>
          密码
          <input type="password" value={password}
                 onChange={e => setPassword(e.target.value)}
                 autoComplete="current-password" required />
        </label>
        {error && <p className="login-error" role="alert">{error}</p>}
        <button type="submit" disabled={busy}>{busy ? '登录中…' : '登录'}</button>
      </form>
    </div>
  );
}

// ==================== 主应用 ====================

function App() {
  // user: undefined=校验中 | null=未登录 | object=已登录
  const [user, setUser] = useState(undefined);
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  // 消息缓存：{convId: [{role:'user'|'ai', content, outcome?, riskNotice?}]}
  const [messageCache, setMessageCache] = useState({});
  const [importableCount, setImportableCount] = useState(0);
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

  // ---- 启动：校验登录态（HttpOnly Cookie 由浏览器自动携带） ----
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { status, data } = await apiFetch('/auth/me');
        if (cancelled) return;
        if (status === 200) setUser(data);
        else setUser(null);
      } catch {
        if (!cancelled) setUser(null);  // 后端不可达时按未登录处理，由登录页报错
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ---- 登录后：拉会话列表 + 探测旧 localStorage 可导入数据 ----
  useEffect(() => {
    if (!user) return;
    (async () => {
      const { status, data } = await apiFetch('/conversations');
      if (status === 200) {
        setConversations(data.conversations || []);
      }
      setImportableCount(loadLegacyConversations().length);
    })();
  }, [user]);

  // activeId 失效（列表为空或记录了已删除的会话）时回退到不选中
  useEffect(() => {
    if (conversations.length && !conversations.some(c => c.id === activeId)) {
      setActiveId(conversations[0].id);
    }
    if (!conversations.length && activeId) setActiveId(null);
  }, [conversations, activeId]);

  // ---- 打开会话：历史懒加载（服务端为唯一事实源） ----
  useEffect(() => {
    if (!activeId || messageCache[activeId]) return;
    let cancelled = false;
    (async () => {
      const { status, data } = await apiFetch(`/conversations/${activeId}/messages`);
      if (cancelled || status !== 200) return;
      setMessageCache(prev => ({
        ...prev,
        [activeId]: (data.messages || []).map(m => ({
          role: m.role === 'assistant' ? 'ai' : 'user',
          content: m.content,
        })),
      }));
    })();
    return () => { cancelled = true; };
  }, [activeId, messageCache]);

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

  // 新消息或加载状态变化时，滚动到对话区底部
  useEffect(() => {
    const el = chatBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messageCache, activeId, loading]);

  // 卸载/切换会话时停掉进行中的打字动画
  useEffect(() => () => {
    if (streamRef.current) clearInterval(streamRef.current.timer);
  }, [activeId]);

  const active = conversations.find(c => c.id === activeId) || null;
  const messages = (activeId && messageCache[activeId]) || [];

  const patchMessages = useCallback((convId, updater) => {
    setMessageCache(prev => ({
      ...prev,
      [convId]: updater(prev[convId] || []),
    }));
  }, []);

  // ---- 流式渲染：把完整回答逐段显示（后端是同步返回，这里做前端渐进呈现） ----
  const streamAnswer = useCallback((full, onDone) => {
    if (streamRef.current) clearInterval(streamRef.current.timer);
    let i = 0;
    const step = Math.max(4, Math.round(full.length / 120)); // ~120 次刷完
    const state = { timer: null, full };
    state.timer = setInterval(() => {
      i += step;
      const visible = full.slice(0, i);
      patchMessages(activeId, msgs => {
        const next = [...msgs];
        const last = next[next.length - 1];
        if (last && last.role === 'ai') next[next.length - 1] = { ...last, content: visible };
        return next;
      });
      if (i >= full.length) {
        clearInterval(state.timer);
        streamRef.current = null;
        onDone && onDone();
      }
    }, 24);
    streamRef.current = state;
  }, [patchMessages, activeId]);

  // ---- 发送消息 -> 调用后端 /agent/chat（主 Agent 编排链路，同步返回完整回答） ----
  const sendChat = (text) => {
    const question = (text ?? message).trim();
    if (!question || loading || !activeId) return; // 防止连点
    if (question.length > MAX_QUESTION_LEN) {
      patchMessages(activeId, msgs => [...msgs,
        { role: "ai", content: `问题过长（${MAX_QUESTION_LEN} 字以内），请精简后再发。`, outcome: "unclassified" }]);
      return;
    }
    setMessage("");
    setLoading(true);

    // 乐观渲染：立即上屏用户消息与占位的助手消息
    patchMessages(activeId, msgs => [...msgs,
      { role: "user", content: question },
      { role: "ai", content: "" }]);
    setConversations(prev => prev.map(c =>
      c.id === activeId
        ? { ...c, title: (c.title === '新的对话' && (messageCache[activeId] || []).length === 0)
            ? question.slice(0, 18) : c.title, updated_at: Date.now() }
        : c));

    (async () => {
      try {
        const { status, data } = await apiFetch('/agent/chat', {
          method: 'POST', body: { session_id: activeId, question },
        });
        if (status !== 200) {
          patchMessages(activeId, msgs => {
            const next = [...msgs];
            const last = next[next.length - 1];
            if (last && last.role === 'ai') {
              next[next.length - 1] = { ...last, content: errText(data, '服务暂时不可用，请稍后重试。') };
            }
            return next;
          });
          setLoading(false);
          return;
        }
        // 后端已把正文、风险提示、来源、免责声明组装进 answer；非 200 时接口返回
        // {detail}（固定话术，不含内部信息），故优先取 answer，其次 detail。
        const content = data.answer || data.detail || "服务暂时不可用，请稍后重试。";
        patchMessages(activeId, msgs => {
          const next = [...msgs];
          const last = next[next.length - 1];
          if (last && last.role === 'ai') {
            next[next.length - 1] = {
              ...last,
              outcome: data.outcome || null,
              riskNotice: data.risk_notice || null,
            };
          }
          return next;
        });
        streamAnswer(content, () => setLoading(false));
      } catch {
        patchMessages(activeId, msgs => {
          const next = [...msgs];
          const last = next[next.length - 1];
          if (last && last.role === 'ai') {
            next[next.length - 1] = { ...last, content: "网络请求失败，请检查后端服务。" };
          }
          return next;
        });
        setLoading(false);
      }
    })();
  };

  // ---- 处理文件上传函数（管理员功能；未授权时后端返回 403） ----
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        credentials: 'include',
        headers: { ...CSRF_HEADERS },  // FormData 不能手动设 Content-Type
        body: formData,
      });
      const data = await response.json();
      // /upload 解析/入库失败时后端仍返回 HTTP 200，失败信息在 message 里
      // （已记录的口径缝隙）：前端按措辞识别，避免把失败误报成成功。
      const text = data.message || "上传成功";
      const failed = !response.ok || text.includes("处理失败");
      patchMessages(activeId || '', msgs => [...msgs, { role: "ai", content: text, outcome: failed ? "retrieval_failed" : null }]);
    } catch (error) {
      console.error("上传失败:", error);
      patchMessages(activeId || '', msgs => [...msgs, { role: "ai", content: "上传失败，请检查后端服务", outcome: "retrieval_failed" }]);
    } finally {
      setUploading(false);
      e.target.value = ''; // 清空 input 的值，允许再次上传同一个文件
    }
  };

  // ---- 新建会话：服务端生成 id（前端不再自造 session_id） ----
  const startNewChat = async () => {
    const { status, data } = await apiFetch('/conversations', {
      method: 'POST', body: { title: '新的对话' },
    });
    if (status !== 200) return;
    setConversations(prev => [{ ...data, updated_at: Date.now() }, ...prev]);
    setActiveId(data.id);
    setSidebarOpen(false);
  };

  // ---- 删除会话：后端软删 + 级联清运行时状态，前端同步移除 ----
  const deleteConversation = async (id) => {
    await apiFetch(`/conversations/${id}`, { method: 'DELETE' });
    setConversations(prev => prev.filter(c => c.id !== id));
    setMessageCache(prev => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  // ---- 一次性导入旧 localStorage 历史（决策 #10） ----
  const importLegacy = async () => {
    const legacy = loadLegacyConversations();
    if (!legacy.length) { setImportableCount(0); return; }
    const { status } = await apiFetch('/conversations/import', {
      method: 'POST',
      body: {
        conversations: legacy.map(c => ({
          title: c.title,
          created_at: (c.createdAt || Date.now()) / 1000,
          messages: c.messages.map(m => ({ role: m.role === 'ai' ? 'assistant' : 'user', content: m.content })),
        })),
      },
    });
    if (status === 200) {
      clearLegacyConversations();
      setImportableCount(0);
      const list = await apiFetch('/conversations');
      if (list.status === 200) setConversations(list.data.conversations || []);
    }
  };

  const logout = async () => {
    await apiFetch('/auth/logout', { method: 'POST' });
    setUser(null);
    setConversations([]);
    setMessageCache({});
    setActiveId(null);
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

  if (user === undefined) {
    return <div className="app-shell loading-screen">加载中…</div>;
  }
  if (user === null) {
    return <LoginScreen onLogin={setUser} />;
  }

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
        <div className="sidebar-footer">
          <span className="sidebar-user" title={user.username}>{user.display_name || user.username}</span>
          <button className="logout-btn" onClick={logout}>退出登录</button>
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
          {importableCount > 0 && (
            <div className="import-banner">
              检测到本机保存的 {importableCount} 段旧版对话记录，是否导入到当前账号？
              <button onClick={importLegacy}>导入</button>
              <button onClick={() => { clearLegacyConversations(); setImportableCount(0); }}>忽略</button>
            </div>
          )}
          <div className="chat-box" ref={chatBoxRef}>
            {!active && (
              <div className="welcome">
                <div className="welcome-icon">🩺</div>
                <p>您好，我是医疗科普问答助手</p>
                <p className="welcome-sub">请描述您的健康问题，我会依据知识库资料回答；
                  也可以问我药品用法、检查指标、疾病科普等内容。</p>
                <button className="suggestion-chip start-chat" onClick={startNewChat}>开始新对话</button>
              </div>
            )}
            {active && messages.length === 0 && (
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

          {active && (
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
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
