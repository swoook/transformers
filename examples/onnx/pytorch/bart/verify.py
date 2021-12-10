from transformers import BartTokenizer, BartModel
import onnxruntime as ort


def main():
    tokenizer = BartTokenizer.from_pretrained(paths['model(torch)'])


if __name__ == '__main__':
	main()