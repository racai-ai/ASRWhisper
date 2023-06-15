
def test_loader(loader, wtokenizer):
    print("Testing the loader:")
    for b in loader:
        print(b["labels"].shape)
        print(b["input_ids"].shape)
        print(b["dec_input_ids"].shape)

        for token, dec in zip(b["labels"], b["dec_input_ids"]):
            token[token == -100] = wtokenizer.eot
            text = wtokenizer.decode(token, skip_special_tokens=False)
            print(text)

            dec[dec == -100] = wtokenizer.eot
            text = wtokenizer.decode(dec, skip_special_tokens=False)
            print(text)


        break
