import section1

def main():
    print("----- Section 1 -----")
    section1.s1("../data/cat"+".jpg")
    section1.s1("../data/cat2" + ".CR2")
    section1.s1("../data/dog" + ".jpg")
    section1.s1("../data/dog2"+".CR2")

    print("--- Question 4 ---")
    section1.s1_4("../data/cat"+".jpg")
    section1.s1_4("../data/cat2" + ".CR2")
    section1.s1_4("../data/dog" + ".jpg")
    section1.s1_4("../data/dog2" + ".CR2")


if __name__ == "__main__":
    main()
