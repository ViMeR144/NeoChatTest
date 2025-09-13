-- Neural Network Database Initialization
-- PostgreSQL schema for the Advanced Neural Network system

-- Create database if not exists (this will be handled by Docker)
-- CREATE DATABASE neural_db;

-- Connect to the database
\c neural_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS neural;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO neural, public;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    generation_time_ms INTEGER,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    temperature DECIMAL(3,2),
    top_k INTEGER,
    top_p DECIMAL(3,2),
    repetition_penalty DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Generation requests table
CREATE TABLE IF NOT EXISTS generation_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    response_text TEXT,
    tokens_generated INTEGER,
    generation_time_ms INTEGER,
    model_info JSONB,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Model configurations table
CREATE TABLE IF NOT EXISTS model_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    parameters BIGINT,
    context_length INTEGER,
    max_tokens INTEGER,
    default_temperature DECIMAL(3,2) DEFAULT 0.8,
    default_top_k INTEGER DEFAULT 50,
    default_top_p DECIMAL(3,2) DEFAULT 0.9,
    default_repetition_penalty DECIMAL(3,2) DEFAULT 1.1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    permissions JSONB DEFAULT '["read", "write"]'::jsonb,
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Usage statistics table
CREATE TABLE IF NOT EXISTS usage_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    requests_count INTEGER DEFAULT 0,
    tokens_generated BIGINT DEFAULT 0,
    generation_time_ms BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date)
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_content_gin ON messages USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_generation_requests_user_id ON generation_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_generation_requests_conversation_id ON generation_requests(conversation_id);
CREATE INDEX IF NOT EXISTS idx_generation_requests_status ON generation_requests(status);
CREATE INDEX IF NOT EXISTS idx_generation_requests_created_at ON generation_requests(created_at);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);

CREATE INDEX IF NOT EXISTS idx_usage_stats_user_id ON usage_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_stats_date ON usage_stats(date);

CREATE INDEX IF NOT EXISTS idx_system_metrics_service ON system_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configs_updated_at BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_usage_stats_updated_at BEFORE UPDATE ON usage_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create functions for analytics
CREATE OR REPLACE FUNCTION get_user_stats(user_uuid UUID, start_date DATE, end_date DATE)
RETURNS TABLE (
    total_requests BIGINT,
    total_tokens BIGINT,
    avg_generation_time DECIMAL,
    total_conversations BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(us.requests_count), 0) as total_requests,
        COALESCE(SUM(us.tokens_generated), 0) as total_tokens,
        COALESCE(AVG(us.generation_time_ms), 0) as avg_generation_time,
        COUNT(DISTINCT c.id) as total_conversations
    FROM users u
    LEFT JOIN usage_stats us ON u.id = us.user_id 
        AND us.date >= start_date 
        AND us.date <= end_date
    LEFT JOIN conversations c ON u.id = c.user_id
        AND c.created_at >= start_date
        AND c.created_at <= end_date + INTERVAL '1 day'
    WHERE u.id = user_uuid
    GROUP BY u.id;
END;
$$ LANGUAGE plpgsql;

-- Create views for common queries
CREATE OR REPLACE VIEW conversation_summary AS
SELECT 
    c.id,
    c.user_id,
    c.title,
    c.created_at,
    c.updated_at,
    COUNT(m.id) as message_count,
    SUM(m.tokens_used) as total_tokens,
    AVG(m.generation_time_ms) as avg_generation_time
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY c.id, c.user_id, c.title, c.created_at, c.updated_at;

CREATE OR REPLACE VIEW user_activity AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.created_at,
    u.last_login,
    COUNT(DISTINCT c.id) as total_conversations,
    COUNT(DISTINCT m.id) as total_messages,
    SUM(m.tokens_used) as total_tokens_used,
    AVG(m.generation_time_ms) as avg_generation_time
FROM users u
LEFT JOIN conversations c ON u.id = c.user_id
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY u.id, u.username, u.email, u.created_at, u.last_login;

-- Insert default model configuration
INSERT INTO model_configs (name, version, description, parameters, context_length, max_tokens, default_temperature, default_top_k, default_top_p, default_repetition_penalty) 
VALUES (
    'Advanced Neural Network',
    '1.0.0',
    'Advanced neural network with dynamic text generation capabilities',
    7000000000,
    8192,
    2048,
    0.8,
    50,
    0.9,
    1.1
) ON CONFLICT (name) DO NOTHING;

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, is_admin) 
VALUES (
    'admin',
    'admin@neural-network.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8V9K8Yz6T.', -- admin123
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA neural TO neural;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA neural TO neural;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA neural TO neural;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA neural TO neural;

-- Create monitoring schema tables
SET search_path TO monitoring, public;

CREATE TABLE IF NOT EXISTS service_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'unhealthy')),
    response_time_ms INTEGER,
    last_check TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS alert_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    message TEXT NOT NULL,
    service_name VARCHAR(100),
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_service_health_service ON service_health(service_name);
CREATE INDEX IF NOT EXISTS idx_service_health_last_check ON service_health(last_check);
CREATE INDEX IF NOT EXISTS idx_alert_history_triggered_at ON alert_history(triggered_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_severity ON alert_history(severity);

-- Reset search path
SET search_path TO neural, public;

-- Create analytics schema tables
SET search_path TO analytics, public;

CREATE TABLE IF NOT EXISTS daily_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    total_requests BIGINT DEFAULT 0,
    total_tokens BIGINT DEFAULT 0,
    avg_response_time_ms DECIMAL(10,2) DEFAULT 0,
    error_rate DECIMAL(5,2) DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, service_name)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date);
CREATE INDEX IF NOT EXISTS idx_daily_metrics_service ON daily_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON performance_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);

-- Reset search path
SET search_path TO neural, public;

-- Create a function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS VOID AS $$
BEGIN
    -- Delete old sessions (older than 30 days)
    DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Delete old system metrics (older than 90 days)
    DELETE FROM system_metrics WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Delete old alert history (older than 180 days)
    DELETE FROM monitoring.alert_history WHERE triggered_at < CURRENT_TIMESTAMP - INTERVAL '180 days';
    
    -- Delete old performance metrics (older than 90 days)
    DELETE FROM analytics.performance_metrics WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Create a function to update usage statistics
CREATE OR REPLACE FUNCTION update_usage_stats()
RETURNS VOID AS $$
BEGIN
    INSERT INTO usage_stats (user_id, date, requests_count, tokens_generated, generation_time_ms)
    SELECT 
        user_id,
        CURRENT_DATE,
        COUNT(*),
        SUM(tokens_generated),
        SUM(generation_time_ms)
    FROM generation_requests
    WHERE created_at >= CURRENT_DATE
        AND status = 'completed'
    GROUP BY user_id
    ON CONFLICT (user_id, date) 
    DO UPDATE SET
        requests_count = usage_stats.requests_count + EXCLUDED.requests_count,
        tokens_generated = usage_stats.tokens_generated + EXCLUDED.tokens_generated,
        generation_time_ms = usage_stats.generation_time_ms + EXCLUDED.generation_time_ms,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get system health
CREATE OR REPLACE FUNCTION get_system_health()
RETURNS TABLE (
    service_name VARCHAR(100),
    status VARCHAR(20),
    response_time_ms INTEGER,
    last_check TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sh.service_name,
        sh.status,
        sh.response_time_ms,
        sh.last_check
    FROM monitoring.service_health sh
    ORDER BY sh.last_check DESC;
END;
$$ LANGUAGE plpgsql;

-- Final setup
COMMENT ON DATABASE neural_db IS 'Advanced Neural Network Database';
COMMENT ON SCHEMA neural IS 'Core neural network application schema';
COMMENT ON SCHEMA monitoring IS 'System monitoring and health checks';
COMMENT ON SCHEMA analytics IS 'Analytics and performance metrics';

-- Grant permissions to monitoring and analytics schemas
GRANT USAGE ON SCHEMA monitoring TO neural;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO neural;
GRANT USAGE ON SCHEMA analytics TO neural;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO neural;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Neural Network Database initialized successfully!';
    RAISE NOTICE 'Default admin user created: admin / admin123';
    RAISE NOTICE 'Default model configuration created: Advanced Neural Network v1.0.0';
END $$;

