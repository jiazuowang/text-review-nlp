# Text Review

**English text-origin analysis · three trained models · local desktop application**

Text Review compares lexical and writing-style features for classifying human-written and machine-generated English text. It connects the experiments to a local application with document import, model comparison, passage-impact inspection, draft comparison, batch processing, and report export.

This group NLP project includes trained models and runtime resources. Install the dependencies to run the application locally.

## Run the application

The original application was tested on Windows with Python 3.14.6. Run these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.app.desktop
```

On macOS/Linux, use `.venv/bin/python` and install Tkinter if needed. Those platforms have not been verified for this release. Once dependencies are installed, inference and report export run locally without sending documents to an external API.

For a command-line example using a newly written demonstration text:

```powershell
.venv\Scripts\python.exe -m src.app.cli examples/demo.txt report.json --model "Compare All"
```

The example illustrates application behavior; it is not a labelled benchmark. The GUI also supports TXT, DOCX and text-based PDF import, CSV batches, and Word/PDF/JSON export. Scanned PDFs require OCR elsewhere.

## How it works

```mermaid
flowchart LR
    A[Text or document] --> B[Input validation]
    B --> C[Saved feature and classifier pipeline]
    C --> D[Classification score]
    D --> E[Selected threshold]
    E --> F[Results and report export]
```

| Model | Representation | Classifier |
| --- | --- | --- |
| Lexical Features | Word TF-IDF | Logistic regression |
| Writing Style | 32 linguistic measurements | Random forest |
| Combined Features | Word and character TF-IDF plus style features | Logistic regression |

Each `pipeline.joblib` stores the fitted preprocessing and classifier together. `src/app/service.py` verifies the configured file hash, loads the selected pipeline, and runs inference. No retraining occurs when a user analyzes a document. See [model details](MODEL_CARD.md).

## Recorded experiments

The original grouped split contained 2,318 training, 492 validation, and 506 test documents. Related prompts and detected duplicates were kept in the same partition. Baseline hyperparameters were selected using validation macro F1.

| Model | Test accuracy | Test macro F1 | Evaluation scope |
| --- | ---: | ---: | --- |
| Lexical Features | 73.91% | 0.7346 | Frozen baseline, 506 documents |
| Writing Style | 70.36% | 0.7015 | Frozen baseline, 506 documents |
| Combined Features | 77.47% | 0.7708 | Follow-up on the already-used test partition |

These results describe the course dataset. The combined model was developed after baseline test evaluation, so its improvement needs confirmation on a fresh test set. Aggregate records are in [results](results/).

The threshold controls the trade-off between false positives and missed detections. Scores express model responses; sentence-deletion highlights show prediction sensitivity. Neither establishes authorship. Evaluation covered 50–1,000-word English documents. See the [model card](MODEL_CARD.md) for interpretation and limits.

## My contribution and team acknowledgements

I proposed the idea and workflow after discussion with my teammates and led the initial prototype. My work covered literature research, dataset preparation, model experiments, application integration, progress reports and a draft demonstration script. I used Codex for development assistance.

My teammates contributed subsequent code and documentation revisions, demonstration preparation and final video production. Individual credits will be added after teammates confirm their preferred public attribution.

## Verification and reproduction

```powershell
.venv\Scripts\python.exe -m scripts.test_public
```

This runs the original test suite, explicitly skipping the one check that requires the excluded frozen training dataset. It includes model inference, input handling, feature checks, and document-export tests. See [verification scope](docs/VERIFICATION.md).

The original training/evaluation source is included for inspection. Exact retraining and the original submission-verification commands require the excluded dataset, manifests, and experiment artifacts; running the desktop app does not. See [reproduction scope](docs/REPRODUCTION.md).

## Sources and use

The experiments use [Beemo](https://github.com/Toloka/beemo) and its [paper](https://aclanthology.org/2025.naacl-long.357/). Style-feature design was informed by [KatFishNet](https://aclanthology.org/2025.acl-long.1030/). The application bundles the NLTK English POS resource and DejaVu font fallbacks with their notices.

See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and reuse conditions. The original text corpus and course submission documents are excluded. Licensing of group contributions requires the contributors' agreement.
