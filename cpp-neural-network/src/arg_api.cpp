#include <iostream>
#include <string>
#include <algorithm>
#include <random>
#include <vector>

int main(int argc, char* argv[]) {
    // Проверяем API режим
    bool api_mode = false;
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--api") {
            api_mode = true;
            break;
        }
    }
    
    if(api_mode) {
        // Простой API режим - получаем сообщение как аргумент
        std::string user_input;
        
        // Ищем аргумент после --api
        for(int i = 1; i < argc; i++) {
            if(std::string(argv[i]) == "--api" && i + 1 < argc) {
                user_input = argv[i + 1];
                break;
            }
        }
        
        // Если нет аргумента, читаем из stdin
        if(user_input.empty()) {
            std::getline(std::cin, user_input);
        }
        
        // Простые ответы
        std::transform(user_input.begin(), user_input.end(), user_input.begin(), ::tolower);
        
        // Приветствия
        if(user_input.find("привет") != std::string::npos || user_input.find("здравствуй") != std::string::npos || user_input.find("добрый день") != std::string::npos || user_input.find("добрый вечер") != std::string::npos || user_input.find("доброе утро") != std::string::npos || user_input.find("салют") != std::string::npos || user_input.find("хай") != std::string::npos) {
            std::cout << "Neural Network: Привет! Как дела? Рад тебя видеть! Как настроение?" << std::endl;
        } 
        // Как дела
        else if(user_input.find("как дела") != std::string::npos || user_input.find("как ты") != std::string::npos || user_input.find("что нового") != std::string::npos || user_input.find("как жизнь") != std::string::npos || user_input.find("как поживаешь") != std::string::npos || user_input.find("как настроение") != std::string::npos) {
            std::cout << "Neural Network: У меня все отлично! Я изучаю новые алгоритмы машинного обучения и улучшаю свои ответы. А у тебя как дела? Что интересного происходит в жизни?" << std::endl;
        } 
        // Благодарность
        else if(user_input.find("спасибо") != std::string::npos || user_input.find("благодарю") != std::string::npos || user_input.find("спс") != std::string::npos || user_input.find("пасиб") != std::string::npos) {
            std::cout << "Neural Network: Пожалуйста! Всегда рад помочь! Если есть еще вопросы - спрашивай!" << std::endl;
        } 
        // Прощание
        else if(user_input.find("пока") != std::string::npos || user_input.find("до свидания") != std::string::npos || user_input.find("увидимся") != std::string::npos || user_input.find("до встречи") != std::string::npos || user_input.find("чао") != std::string::npos || user_input.find("бай") != std::string::npos) {
            std::cout << "Neural Network: До свидания! Было приятно пообщаться! Увидимся в следующий раз!" << std::endl;
        } 
        // О себе
        else if(user_input.find("что ты") != std::string::npos || user_input.find("кто ты") != std::string::npos || user_input.find("расскажи о себе") != std::string::npos || user_input.find("что ты умеешь") != std::string::npos) {
            std::cout << "Neural Network: Я - продвинутая нейронная сеть, написанная на C++! Я умею общаться на русском и английском языках, отвечать на вопросы, поддерживать беседу и постоянно учусь новому. Моя цель - быть полезным собеседником!" << std::endl;
        } 
        // Помощь
        else if(user_input.find("помощь") != std::string::npos || user_input.find("помоги") != std::string::npos || user_input.find("что ты можешь") != std::string::npos || user_input.find("функции") != std::string::npos) {
            std::cout << "Neural Network: Я могу поговорить с тобой на разные темы, ответить на вопросы, поддержать беседу, рассказать о себе и просто поболтать! Попробуй спросить меня о чем угодно - я постараюсь ответить интересно и полезно!" << std::endl;
        }
        // Вопросы о программировании
        else if(user_input.find("программирование") != std::string::npos || user_input.find("код") != std::string::npos || user_input.find("программа") != std::string::npos || user_input.find("алгоритм") != std::string::npos || user_input.find("c++") != std::string::npos) {
            std::cout << "Neural Network: Программирование - это увлекательная область! Я сама написана на C++ и использую различные алгоритмы для обработки речи. Если у тебя есть вопросы по программированию - с удовольствием помогу!" << std::endl;
        }
        // Искусственный интеллект
        else if(user_input.find("искусственный интеллект") != std::string::npos || user_input.find("нейросеть") != std::string::npos || user_input.find("машинное обучение") != std::string::npos || user_input.find("ии") != std::string::npos || user_input.find("ai") != std::string::npos) {
            std::cout << "Neural Network: Искусственный интеллект - это моя стихия! Я постоянно изучаю новые методы машинного обучения, чтобы лучше понимать человеческую речь и давать более точные ответы. Это очень перспективная область!" << std::endl;
        }
        // Погода
        else if(user_input.find("погода") != std::string::npos || user_input.find("дождь") != std::string::npos || user_input.find("солнце") != std::string::npos || user_input.find("снег") != std::string::npos || user_input.find("ветер") != std::string::npos) {
            std::cout << "Neural Network: К сожалению, я не могу проверить актуальную погоду, но могу поговорить о климате в целом! Какую погоду ты предпочитаешь? Я люблю, когда солнечно и тепло - это создает хорошее настроение!" << std::endl;
        }
        // Еда
        else if(user_input.find("еда") != std::string::npos || user_input.find("есть") != std::string::npos || user_input.find("пища") != std::string::npos || user_input.find("кушать") != std::string::npos || user_input.find("обед") != std::string::npos || user_input.find("завтрак") != std::string::npos || user_input.find("ужин") != std::string::npos) {
            std::cout << "Neural Network: Еда - это важная часть нашей жизни! Хотя я не ем в традиционном понимании, я понимаю, как важна хорошая еда для настроения и здоровья. Что ты любишь готовить или есть?" << std::endl;
        }
        // Учеба/работа
        else if(user_input.find("учусь") != std::string::npos || user_input.find("работаю") != std::string::npos || user_input.find("школа") != std::string::npos || user_input.find("университет") != std::string::npos || user_input.find("работа") != std::string::npos || user_input.find("профессия") != std::string::npos) {
            std::cout << "Neural Network: Учеба и работа - это важные части жизни! Я сама постоянно учусь новому. Расскажи, чем ты занимаешься? Учусь ли ты чему-то новому или работаешь над интересными проектами?" << std::endl;
        }
        // Хобби
        else if(user_input.find("хобби") != std::string::npos || user_input.find("увлечение") != std::string::npos || user_input.find("интерес") != std::string::npos || user_input.find("занятие") != std::string::npos || user_input.find("спорт") != std::string::npos || user_input.find("музыка") != std::string::npos || user_input.find("книги") != std::string::npos) {
            std::cout << "Neural Network: Хобби делают жизнь интереснее! У меня есть увлечение - изучение языков и алгоритмов. А что тебе нравится делать в свободное время? Есть ли у тебя интересные хобби?" << std::endl;
        }
        // Английские фразы
        else if(user_input.find("hello") != std::string::npos || user_input.find("hi") != std::string::npos || user_input.find("hey") != std::string::npos) {
            std::cout << "Neural Network: Hello! How are you doing today? Nice to meet you!" << std::endl;
        } else if(user_input.find("how are you") != std::string::npos || user_input.find("how do you do") != std::string::npos) {
            std::cout << "Neural Network: I'm doing great! Learning new algorithms and improving my responses. How about you? What's new in your life?" << std::endl;
        } else if(user_input.find("thank you") != std::string::npos || user_input.find("thanks") != std::string::npos) {
            std::cout << "Neural Network: You're welcome! Always happy to help! Feel free to ask if you have more questions!" << std::endl;
        } else if(user_input.find("goodbye") != std::string::npos || user_input.find("bye") != std::string::npos) {
            std::cout << "Neural Network: Goodbye! It was nice talking to you! See you next time!" << std::endl;
        } else if(user_input.find("what are you") != std::string::npos || user_input.find("who are you") != std::string::npos) {
            std::cout << "Neural Network: I'm an advanced neural network written in C++! I can communicate in Russian and English, answer questions, maintain conversations, and constantly learn new things. My goal is to be a helpful conversational partner!" << std::endl;
        } else if(user_input.find("help") != std::string::npos) {
            std::cout << "Neural Network: I can talk about various topics, answer questions, maintain conversations, tell you about myself, and just chat! Try asking me anything - I'll try to give interesting and useful answers!" << std::endl;
        } 
        // Общие ответы с рандомом
        else {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::vector<std::string> responses = {
                "Интересно! Расскажи мне больше об этом. Я люблю узнавать новое и могу поговорить на самые разные темы - от программирования до философии. Что тебя особенно интересует?",
                "Очень любопытно! Я постоянно учусь и развиваюсь. А что ты думаешь об этом? Хочешь обсудить что-то конкретное?",
                "Отличная тема для разговора! Я готов поддержать беседу на любую тему. Что бы ты хотел обсудить?",
                "Интригующе! Я нейросеть, которая любит общаться. Расскажи, что тебя волнует или интересует?",
                "Захватывающе! Я всегда рад поговорить. Есть ли что-то, о чем ты хотел бы поговорить?",
                "Интересная мысль! Я умею поддерживать разговор на разные темы. Что тебе интересно?",
                "Классно! Я люблю общаться и узнавать новое. О чем бы ты хотел поговорить?",
                "Здорово! Я готов к разговору на любую тему. Что у тебя на уме?",
                "Отлично! Я нейросеть, которая любит общаться. Расскажи, что тебя интересует?",
                "Прекрасно! Я всегда рад поддержать беседу. О чем поговорим?"
            };
            
            std::uniform_int_distribution<> dis(0, responses.size() - 1);
            int randomIndex = dis(gen);
            std::cout << "Neural Network: " << responses[randomIndex] << std::endl;
        }
    } else {
        // Интерактивный режим
        std::cout << "=== Arg C++ Neural Network ===" << std::endl;
        std::cout << "Type 'exit' to quit" << std::endl;
        
        std::string user_input;
        while(true) {
            std::cout << "You: ";
            std::getline(std::cin, user_input);
            
            if(user_input == "exit") {
                std::cout << "Neural Network: Goodbye!" << std::endl;
                break;
            }
            
            std::cout << "Neural Network: I understand your message." << std::endl;
        }
    }
    
    return 0;
}
