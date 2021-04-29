

def main():
    file_path = r"Data/dummy.wtag"
    maxk = 3
    kgrams_maker = lambda s_t, k: zip(*[s_t[i:] for i in range(k)])
    with open(file_path) as f:
        for line in f:
            split_words = line.split(" ")
            for k in range(2, maxk):
                kgrams = kgrams_maker(split_words, k)
                for kgram in kgrams:
                    splitted_list = (word_tag.split("_") for word_tag in kgram)
                    words, tags = list(zip(*splitted_list))
            if (words, tags) not in self.kgram_dict:
                self.kgram_dict[(words, tags)] = 0
            self.kgram_dict[(words, tags)] += 1

def extract_kgrams(self, maxk):
    kgrams_maker = lambda s_t, k: zip(*[s_t[i:] for i in range(k)])
    with open(file_path) as f:
        for line in f:
            split_words = line.split(" ")
            for k in range(2, maxk):
                kgrams = kgrams_maker(split_words, k)
                for kgram in kgrams:
                    splitted_list = (word_tag.split("_") for word_tag in kgram)
                    words, tags = list(zip(*splitted_list))
            if (words, tags) not in self.kgram_dict:
                self.kgram_dict[(words, tags)] = 0
            self.kgram_dict[(words, tags)] += 1

if __name__ == '__main__':
    main()

