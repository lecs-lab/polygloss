# IMTVault

[![CLDF validation](https://github.com/cldf-datasets/imtvault/workflows/CLDF-validation/badge.svg)](https://github.com/cldf-datasets/imtvault/actions?query=workflow%3ACLDF-validation)

[CLDF dataset](cldf/) containing Interlinear Glossed Text extracted from linguistic literature.


## How to cite

If you use data from this dataset, please cite the [released version of the data you are using]() as
well as the [paper introducing IMTVault](http://www.lrec-conf.org/proceedings/lrec2022/workshops/LDL/pdf/2022.ldl2022-1.3.pdf)

> Krämer, Thomas, and Sebastian Nordhoff. 2022. "IMTVault: Extracting and Enriching Low-resource Language Interlinear Glossed Text from Grammatical Descriptions and Typological Survey Articles: Proceedings of The 8th Workshop on Linked Data in Linguistics within the 13th Language Resources and Evaluation Conference." 13th Language Resources and Evaluation Conference lREC 2022, LREC 2022, Marseille, 24.06.2022.


## Coverage

Distribution of examples in IMTVault across the languages of the world:

![](map.jpg?pacific-centered&language-properties=Examples_Count_Log&language-properties-colormaps=viridis&width=20&height=10&padding-left=5&padding-right=5&padding-top=5&padding-bottom=5&format=jpg&markersize=12#cldfviz.map)


## How to use

The dataset provided in the [`cldf` directory](cldf/) is a valid [CLDF dataset](https://cldf.clld.org). Thus, after
looking up the file and column names for standard CLDF tables and properties in [the metadata](cldf/Generic-metadata.json)
or the [README.md](cldf/README.md), you can use any tool capable of reading CSV to poke around the data.

For example, you could use the commandline tools from the [csvkit package](https://csvkit.readthedocs.io/en/latest/)
- to check whether a particular language is represented in the dataset
  ```shell
  $ csvgrep -c Name -m Amele cldf/languages.csv | csvcut -c ID,Name,Examples_Count
  ID,Name,Examples_Count
  amel1241,Amele,4
  ```
- to filter examples based on values for specific properties
  ```shell
  $ csvgrep -c Language_ID -m amel1241 cldf/examples.csv | csvgrep -c"Gloss" -r"food" | csvcut -c ID,Primary_Text,Translated_Text
  ID,Primary_Text,Translated_Text
  langsci220-e5ca0880e8,[Ija sab fajec nu] huga.,I came to buy food.
  glossa5188-47,[ Ege humeb ] sab josi,We came and they two ate the food.
  ```

If you found suitable examples, you might render them in a human-readable format using [cldfviz.text](https://github.com/cldf/cldfviz).
E.g. the CLDF markdown snippet
```
[](ExampleTable?with_internal_ref_link#cldf:langsci220-e5ca0880e8) 
[](ExampleTable?with_internal_ref_link#cldf:glossa5188-47)

[References](Source?cited_only#cldf:__all__)
```
in a file `amele_examples.md`
would render via
```shell
cldfbench cldfviz.text --text-file amele_examples.md cldf/Generic-metadata.json
```
as

---

> (langsci220-e5ca0880e8) Amele ([Roberts 1987](#source-roberts:87:1), [Schmidtke-Bode et al. 2018](#source-schmidtke-bode:levshina:etal:ed:18): via)
<pre>
[Ija  sab   faj-ec        nu]  h-ug-a.  
1SG   food  buy-INF/NMLZ  for  come-1SG-PST  
‘I came to buy food.’</pre>
 
> (glossa5188-47) Amele ([Stirling 1993](#source-stirling:93): 213, [Bárány and Nikolaeva 2019](#source-barany:nikolaeva:19): via:49)
<pre>
[ Ege   h-u-me-b          ] sab    jo-si-a.  
1PL  come-PRED-SS-1PL     food  eat-3DU.TODPST  
‘We came and they two ate the food.’</pre>
 - <a id="source-barany:nikolaeva:19"> </a>Bárány, András and Nikolaeva, Irina. 2019. Possessors in switch-reference. Glossa: a journal of general linguistics 4(1). Open Library of Humanities.
- <a id="source-roberts:87:1"> </a>Roberts, John. 1987. Amele. London: Croom Helm.
- <a id="source-schmidtke-bode:levshina:etal:ed:18"> </a>Schmidtke-Bode, Karsten and Levshina, Natalia and Michaelis, Susanne Maria and Seržant, Ilja (eds.) 2018. Explanation in typology: Diachronic sources, functional motivations and the nature of the evidence. (Conceptual Foundations of Language Science, 3.) Berlin: Language Science Press.
- <a id="source-stirling:93"> </a>Stirling, Lesley. 1993. Switch-reference and discourse representation. Cambridge: Cambridge University Press.

---