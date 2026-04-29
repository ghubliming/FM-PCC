see 
ipynbs/Dataset_Extraction/(D3IL_dataset_Extraction)CheckP_1_Colab_GPU_FMPCC_MuJoCo.ipynb

---

%%bash
DATASET_DIR="/content/drive/MyDrive/DPCC/dpcc/d3il/environments/dataset/data" # Notice Here We use the ZIP from DPCC folder
AVOIDING_DATA="/content/drive/MyDrive/FMPCC/FM-PCC/d3il/environments/dataset/data/avoiding/data"
ZIP_FILE="$DATASET_DIR/dataset.zip"

echo "========================================="
echo " D3IL DATASET SETUP"
echo " Source: ALRhub/d3il README.md"
echo " Link: https://drive.google.com/file/d/1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8"
echo "========================================="
echo ""

---


Example Code and output (Expected)

---

%%bash
ZIP="/content/drive/MyDrive/DPCC/dpcc/d3il/environments/dataset/data/dataset.zip"

echo "=== Top-level folders in zip ==="
unzip -l "$ZIP" | awk '{print $4}' | cut -d'/' -f1 | sort -u

echo ""
echo "=== File count per folder ==="
unzip -l "$ZIP" | awk '{print $4}' | cut -d'/' -f1 | sort | uniq -c | sort -rn

echo ""
echo "=== Total uncompressed size ==="
unzip -l "$ZIP" | tail -1

```
=== Top-level folders in zip ===

----
aligning
avoiding
inserting
Name
pushing
sorting
stacking

=== File count per folder ===
2638779 sorting
1377112 stacking
 392768 aligning
   2002 pushing
    802 inserting
     99 avoiding
      3
      1 Name
      1 ----

=== Total uncompressed size ===
17689115292                     4411562 files