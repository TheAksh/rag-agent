import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


def main():
    print("Hello from rag-agent!")


if __name__ == "__main__":
    main()
