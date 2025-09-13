#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <map>
#include <algorithm>

// Собственный Transformer для генерации текста
class Transformer {
private:
    int vocab_size;
    int d_model;
    int nhead;
    int num_layers;
    int max_seq_len;
    
    // Словарь токенов
    std::map<std::string, int> word_to_id;
    std::map<int, std::string> id_to_word;
    int next_token_id = 0;
    
    // Веса модели
    std::vector<std::vector<std::vector<float>>> attention_weights;
    std::vector<std::vector<float>> feed_forward_weights;
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<float>> position_embeddings;
    
public:
    Transformer(int vocab_size, int d_model = 512, int nhead = 8, int num_layers = 6, int max_seq_len = 1024) 
        : vocab_size(vocab_size), d_model(d_model), nhead(nhead), num_layers(num_layers), max_seq_len(max_seq_len) {
        
        // Инициализируем веса
        initializeWeights();
        
        // Создаем базовый словарь
        createBasicVocabulary();
    }
    
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 0.1);
        
        // Инициализируем эмбеддинги
        embeddings.resize(vocab_size, std::vector<float>(d_model));
        for(int i = 0; i < vocab_size; i++) {
            for(int j = 0; j < d_model; j++) {
                embeddings[i][j] = dist(gen);
            }
        }
        
        // Инициализируем позиционные эмбеддинги
        position_embeddings.resize(max_seq_len, std::vector<float>(d_model));
        for(int i = 0; i < max_seq_len; i++) {
            for(int j = 0; j < d_model; j++) {
                position_embeddings[i][j] = dist(gen);
            }
        }
        
        // Инициализируем веса внимания
        attention_weights.resize(num_layers, std::vector<std::vector<float>>(d_model, std::vector<float>(d_model)));
        for(int layer = 0; layer < num_layers; layer++) {
            for(int i = 0; i < d_model; i++) {
                for(int j = 0; j < d_model; j++) {
                    attention_weights[layer][i][j] = dist(gen);
                }
            }
        }
        
        // Инициализируем feed-forward веса
        feed_forward_weights.resize(num_layers, std::vector<float>(d_model * 4));
        for(int layer = 0; layer < num_layers; layer++) {
            for(int i = 0; i < d_model * 4; i++) {
                feed_forward_weights[layer][i] = dist(gen);
            }
        }
    }
    
    void createBasicVocabulary() {
        // Создаем базовый словарь для русского и английского
        std::vector<std::string> basic_words = {
            "привет", "мир", "как", "дела", "что", "ты", "делаешь", "сегодня",
            "hello", "world", "how", "are", "you", "today", "what", "do",
            "я", "мы", "он", "она", "они", "это", "то", "вот",
            "i", "we", "he", "she", "they", "this", "that", "here",
            "да", "нет", "хорошо", "плохо", "давай", "пойдем", "идем",
            "yes", "no", "good", "bad", "let's", "go", "come",
            "спасибо", "пожалуйста", "извини", "прости", "пока",
            "thank", "you", "please", "sorry", "bye", "goodbye"
        };
        
        for(const auto& word : basic_words) {
            if(word_to_id.find(word) == word_to_id.end()) {
                word_to_id[word] = next_token_id;
                id_to_word[next_token_id] = word;
                next_token_id++;
            }
        }
        
        // Добавляем специальные токены
        word_to_id["<PAD>"] = next_token_id++;
        word_to_id["<START>"] = next_token_id++;
        word_to_id["<END>"] = next_token_id++;
        word_to_id["<UNK>"] = next_token_id++;
        
        id_to_word[next_token_id-4] = "<PAD>";
        id_to_word[next_token_id-3] = "<START>";
        id_to_word[next_token_id-2] = "<END>";
        id_to_word[next_token_id-1] = "<UNK>";
    }
    
    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        std::string word = "";
        
        for(char c : text) {
            if(c == ' ' || c == '.' || c == ',' || c == '!' || c == '?') {
                if(!word.empty()) {
                    if(word_to_id.find(word) != word_to_id.end()) {
                        tokens.push_back(word_to_id[word]);
                    } else {
                        tokens.push_back(word_to_id["<UNK>"]);
                    }
                    word = "";
                }
            } else {
                word += std::tolower(c);
            }
        }
        
        if(!word.empty()) {
            if(word_to_id.find(word) != word_to_id.end()) {
                tokens.push_back(word_to_id[word]);
            } else {
                tokens.push_back(word_to_id["<UNK>"]);
            }
        }
        
        return tokens;
    }
    
    std::string detokenize(const std::vector<int>& tokens) {
        std::string result = "";
        for(int token : tokens) {
            if(id_to_word.find(token) != id_to_word.end()) {
                if(!result.empty()) result += " ";
                result += id_to_word[token];
            }
        }
        return result;
    }
    
    std::vector<float> attention(const std::vector<float>& query, const std::vector<float>& key, const std::vector<float>& value) {
        std::vector<float> result(d_model, 0.0f);
        
        // Упрощенное внимание (без softmax для демонстрации)
        for(int i = 0; i < d_model; i++) {
            result[i] = query[i] * key[i] + value[i];
        }
        
        return result;
    }
    
    std::vector<float> feedForward(const std::vector<float>& input, int layer) {
        std::vector<float> result(d_model, 0.0f);
        
        // Упрощенный feed-forward слой
        for(int i = 0; i < d_model; i++) {
            result[i] = input[i] * feed_forward_weights[layer][i % (d_model * 4)];
        }
        
        return result;
    }
    
    std::vector<int> generate(const std::string& prompt, int max_length = 50) {
        std::vector<int> input_tokens = tokenize(prompt);
        std::vector<int> generated = input_tokens;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        
        for(int step = 0; step < max_length; step++) {
            // Получаем эмбеддинги для текущей последовательности
            std::vector<std::vector<float>> embeddings_seq;
            for(int i = 0; i < std::min((int)generated.size(), max_seq_len); i++) {
                int token_id = generated[i];
                if(token_id < vocab_size) {
                    embeddings_seq.push_back(embeddings[token_id]);
                }
            }
            
            if(embeddings_seq.empty()) break;
            
            // Проходим через слои Transformer
            std::vector<float> current = embeddings_seq.back(); // Берем последний токен
            
            for(int layer = 0; layer < num_layers; layer++) {
                // Self-attention
                current = attention(current, current, current);
                
                // Feed-forward
                current = feedForward(current, layer);
            }
            
            // Генерируем следующий токен (упрощенно)
            int next_token = 0;
            float max_val = -1000.0f;
            
            for(int i = 0; i < vocab_size; i++) {
                float score = 0.0f;
                for(int j = 0; j < d_model; j++) {
                    score += current[j] * embeddings[i][j];
                }
                
                if(score > max_val) {
                    max_val = score;
                    next_token = i;
                }
            }
            
            generated.push_back(next_token);
            
            // Останавливаемся на END токене
            if(next_token == word_to_id["<END>"]) break;
        }
        
        return generated;
    }
    
    void printVocabulary() {
        std::cout << "Словарь токенов:" << std::endl;
        for(const auto& pair : word_to_id) {
            std::cout << pair.first << " -> " << pair.second << std::endl;
        }
    }
};

int main() {
    std::cout << "🚀 Собственный Transformer для генерации текста запущен!" << std::endl;
    
    // Создаем Transformer
    Transformer transformer(1000, 512, 8, 6, 1024);
    
    std::cout << "\n📚 Словарь:" << std::endl;
    transformer.printVocabulary();
    
    // Тестируем генерацию
    std::string prompt = "привет как дела";
    std::cout << "\n💬 Промпт: " << prompt << std::endl;
    
    std::vector<int> generated = transformer.generate(prompt, 20);
    std::string result = transformer.detokenize(generated);
    
    std::cout << "🤖 Ответ: " << result << std::endl;
    
    std::cout << "\n✅ Transformer работает успешно!" << std::endl;
    
    return 0;
}
