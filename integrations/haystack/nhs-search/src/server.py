import streamlit
import streamlit.cli
import sys

from doc_search import app


def main():
    if streamlit._is_running_with_streamlit:
        app.main()
    else:
        sys.argv = ["streamlit", "run", *sys.argv]
        sys.exit(streamlit.cli.main())


if __name__ == "__main__":
    main()