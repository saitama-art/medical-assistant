from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = (
    "postgresql://postgres:password@localhost:5432/"
    "agent_db"
)

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 初始化所需的数据库表

    app = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {"thread_id": "thread-001"}
    }

    result = app.invoke(
        {"messages": [HumanMessage(content="继续刚才的对话")]},
        config=config
    )