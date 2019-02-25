class Jacky_Chatbot:

    all_setence_list = []
    all_world_list_no_repeat = ''
    temp_combine_odd_setence = ''
    odd_setence_list = []

    wrod_dict_input = {}  # 放使用者輸入文字的字向量陣列
    wrod_dict_data = {}  # 放文本文字的字向量陣列

    def setFile(self, file):
        mfile = open(file, 'r')

        global all_setence_list
        all_setence_list = mfile.readlines()#讀入文本內的所有句字，並存放到一個陣列裡
        # print(all_setence_list)


        #只保留偶數句(使用者說的話)
        global odd_setence_list
        odd_setence_list = all_setence_list.copy()
        for i in range(len(odd_setence_list)-1, 0, -2):
            if i % 2 == 1 and i > 0:
                # print(i)
                del odd_setence_list[i]
        # print(odd_setence_list)


        #拿掉偶數句(使用者說的話)裡每句的換行符號
        for i in range(0, len(odd_setence_list)):
            odd_setence_list[i] = odd_setence_list[i].replace('\n', '')
        # print(odd_setence_list)


        #把偶數句(使用者說的話)轉成一個文字陣列，統計有哪些字(不重複)
        global temp_combine_odd_setence
        temp_combine_odd_setence = ''
        for i in odd_setence_list:
            temp_combine_odd_setence += i
        # print(temp_combine_odd_setence)
        #
        global all_world_list_no_repeat
        all_world_list_no_repeat = list(set([n for n in temp_combine_odd_setence if n!= '-' and n!= ' ' and n!= '\n']))
        # print(all_world_list_no_repeat)



        #把文字陣列內的每個元素當成是 wrod_dict_input 和 wrod_dict_data 的索引
        for i in range(0, len(all_world_list_no_repeat)):
            self.wrod_dict_input[all_world_list_no_repeat[i]] = 0
            self.wrod_dict_data[all_world_list_no_repeat[i]] = 0
        # print(wrod_dict_input)
        # print(wrod_dict_data)


    def caculate_cosine_similarly(self, user_vector, textdata_vector):
        user_vector_length = 0
        textdata_vector_length = 0
        inner_product = 0 #兩向量的內積值

        # 計算使用者輸入向量長度
        for i in user_vector:
            user_vector_length += user_vector[i]**2
        #
        user_vector_length = user_vector_length**0.5
        # print(user_vector_length)

        # 計算文本句子向量長度
        for i in textdata_vector:
            textdata_vector_length += textdata_vector[i]**2
        #
        textdata_vector_length = textdata_vector_length**0.5
        # print(textdata_vector_length)

        #計算兩個向量的內積
        for i in textdata_vector:
            # print(i, '======')
            inner_product += textdata_vector[i]*user_vector[i]

        cosine = inner_product/(user_vector_length*textdata_vector_length)

        # print('cosine = ', cosine)
        return cosine






    #把文本的句子轉成文字向量(wrod_dict_data)
    def conver_sentence_to_vector(self, sentence):
        # 清除掉wrod_dict_data的內容
        for i in range(0, len(all_world_list_no_repeat)):
            self.wrod_dict_data[all_world_list_no_repeat[i]] = 0

        text = sentence
        # print(text)
        for i in text:
            if i in temp_combine_odd_setence:
                self.wrod_dict_data[i] += 1
        # print(wrod_dict_data)


    def chat_with_chatbot(self, txt):

        check_user_input = 0 #紀錄使用者輸入的文字有沒有在文本字典裡，沒有的話就會一直是0

        # 清除掉wrod_dict_input的內容
        for i in range(0, len(all_world_list_no_repeat)):
            self.wrod_dict_input[all_world_list_no_repeat[i]] = 0

        #把使用者輸入的句子轉成文字向量(wrod_dict_input)
        text = txt
        for i in text:
            if i in temp_combine_odd_setence:
                self.wrod_dict_input[i] += 1
                check_user_input += 1
        # print(wrod_dict_input)

        if check_user_input == 0:
            # print('回應結果：抱歉我不知到你在說什麼')
            return '抱歉我不知到你在說什麼'


        # 計算文本內的所有偶數句和使用者輸入句子的相似度
        index = 0
        result_sentence_index = 0
        similarity = 0
        for i in odd_setence_list:
            # print(index)

            self.conver_sentence_to_vector(i)

            temp = self.caculate_cosine_similarly(self.wrod_dict_input, self.wrod_dict_data)

            if  temp > similarity:
                similarity = temp
                result_sentence_index = 2*index+1

            index += 1

        # print('回應結果：', all_setence_list[result_sentence_index])
        return all_setence_list[result_sentence_index]




