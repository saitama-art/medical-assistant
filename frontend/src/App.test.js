import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

// react-markdown 为 ESM；本套测试验证 App 数据流，不复测第三方 Markdown 解析器。
jest.mock('react-markdown', () => ({
  __esModule: true,
  default: ({ children }) => <div data-testid="markdown">{children}</div>,
}));

const jsonResponse = (status, data) => Promise.resolve({
  status,
  json: () => Promise.resolve(data),
});

function mockLoginPage(loginImpl) {
  global.fetch.mockImplementation((url) => {
    if (url.endsWith('/auth/me')) return jsonResponse(401, {});
    if (url.endsWith('/auth/login')) return loginImpl();
    return Promise.reject(new Error(`unexpected request: ${url}`));
  });
}

beforeEach(() => {
  window.localStorage.clear();
  window.matchMedia = jest.fn().mockReturnValue({
    matches: false,
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
  });
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.clearAllMocks();
});

test('shows login page and rejects invalid credentials', async () => {
  mockLoginPage(() => jsonResponse(401, { detail: '用户名或密码错误。' }));
  render(<App />);

  fireEvent.change(await screen.findByLabelText('用户名'), { target: { value: 'demo' } });
  fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'wrong-password' } });
  fireEvent.click(screen.getByRole('button', { name: '登录' }));

  expect(await screen.findByRole('alert')).toHaveTextContent('用户名或密码错误');
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/auth/login'),
    expect.objectContaining({ method: 'POST', credentials: 'include' }),
  );
});

test('shows network error when login request fails', async () => {
  mockLoginPage(() => Promise.reject(new Error('network down')));
  render(<App />);

  fireEvent.change(await screen.findByLabelText('用户名'), { target: { value: 'demo' } });
  fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'demo1234' } });
  fireEvent.click(screen.getByRole('button', { name: '登录' }));

  expect(await screen.findByRole('alert')).toHaveTextContent('网络请求失败，请检查后端服务');
});

test('loads owned conversations and answer history', async () => {
  global.fetch.mockImplementation((url) => {
    if (url.endsWith('/auth/me')) {
      return jsonResponse(200, { id: 1, username: 'demo', display_name: '', is_admin: false });
    }
    if (url.endsWith('/conversations')) {
      return jsonResponse(200, { conversations: [
        { id: 'conv-1', title: '高血压生活方式', created_at: 1, updated_at: 1 },
      ] });
    }
    if (url.endsWith('/conversations/conv-1/messages')) {
      return jsonResponse(200, { messages: [
        { role: 'user', content: '高血压要注意什么？' },
        { role: 'assistant', content: '回答正文\n\n【参考来源】\n- **高血压手册**' },
      ] });
    }
    return Promise.reject(new Error(`unexpected request: ${url}`));
  });

  render(<App />);

  expect(await screen.findByText('高血压生活方式')).toBeInTheDocument();
  expect(await screen.findByText('高血压要注意什么？')).toBeInTheDocument();
  expect(await screen.findByText(/高血压手册/)).toBeInTheDocument();
  expect(screen.getByText('demo')).toBeInTheDocument();
});

test.each([
  ['clarification_required', '需要澄清'],
  ['processing_failed', '处理失败'],
  ['safety_rejected', '安全拦截'],
])('renders backend answer and %s outcome badge', async (outcome, label) => {
  global.fetch.mockImplementation((url) => {
    if (url.endsWith('/auth/me')) {
      return jsonResponse(200, { id: 1, username: 'demo', display_name: '', is_admin: false });
    }
    if (url.endsWith('/conversations')) {
      return jsonResponse(200, { conversations: [
        { id: 'conv-1', title: '新的对话', created_at: 1, updated_at: 1 },
      ] });
    }
    if (url.endsWith('/conversations/conv-1/messages')) {
      return jsonResponse(200, { messages: [] });
    }
    if (url.endsWith('/agent/chat')) {
      return jsonResponse(200, {
        answer: '请补充具体药名。', outcome, sources: [], risk_notice: null,
      });
    }
    return Promise.reject(new Error(`unexpected request: ${url}`));
  });

  render(<App />);
  fireEvent.click(await screen.findByText('高血压患者在生活方式上要注意什么？'));

  expect(await screen.findByText(label)).toBeInTheDocument();
  expect(await screen.findByText('请补充具体药名。')).toBeInTheDocument();
});
