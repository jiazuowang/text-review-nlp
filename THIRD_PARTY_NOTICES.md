# Third-party notices

## Dataset and trained-model provenance

Training used [Beemo](https://github.com/Toloka/beemo), fixed revision `cca3b817eb6d0bf7a60f885f3022696553eabb63`.

Beemo's repository license is MIT (see `docs/sources/BEEMO_LICENSE.txt`), but its README identifies separate conditions for dataset components: No Robots prompts and human-written text are CC-BY-NC-4.0; generated and edited text can be subject to the originating model's terms. The repository-level MIT notice is not a blanket license for every underlying text or downstream use.

This release does not distribute the raw or processed text corpus, individual document predictions, or saved semantic-review annotations. Trained artifacts are provided with this provenance for research/portfolio review; no unrestricted commercial-use claim is made. Obtain source data from upstream and review the applicable conditions before reuse.

Reference: Artemova et al., *Beemo: Benchmark of Expert-edited Machine-generated Outputs*, NAACL 2025, https://aclanthology.org/2025.naacl-long.357/.

## Methods

The style-feature design was informed by Park et al., *KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis*, ACL 2025, https://aclanthology.org/2025.acl-long.1030/. The project adapts selected ideas to English; it does not distribute that paper or claim its algorithm as an original invention.

## Runtime resources

- NLTK English averaged-perceptron POS resource: notices in `docs/sources/NLTK_POS_LICENSE.txt`, `NLTK_DATA_LICENSE.txt`, and `NLTK_LICENSE_OVERVIEW.md`; resource hashes in `resources/pos/manifest.json`.
- DejaVu Serif regular/bold fonts: notice in `resources/fonts/LICENSE_DEJAVU`.
- Python dependencies are installed separately under their own licenses. No virtual environment or third-party package installation is bundled.

## Project contributions

See the README for individual work and team acknowledgements. Contributor names will be added after preferences are confirmed. Licensing of group contributions requires the contributors' agreement; upstream resources retain their own terms.
