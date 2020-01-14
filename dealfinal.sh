rm -rf ../data/JPEGImages ../data/Annotations
mkdir ../data/JPEGImages ../data/Annotations


mv *.jpg ../data/JPEGImages
mv *.xml ../data/Annotations
#mv *.pdf.png ./JPEGImages
rm -rf *.json *.out


#rm -rf ~/Desktop/deconv/data/synthetic/Generate_pdf/JPEGImages ~/Desktop/deconv/data/synthetic/Generate_pdf/Annotations
#mkdir ~/Desktop/deconv/data/synthetic/Generate_pdf/JPEGImages ~/Desktop/deconv/data/synthetic/Generate_pdf/Annotations
#
#
#mv *.jpg ~/Desktop/deconv/data/synthetic/Generate_pdf/JPEGImages
#mv *.xml ~/Desktop/deconv/data/synthetic/Generate_pdf/Annotations
#rm -rf *.aux *.log *.tex *.json *.out  *.png *.pdf

#rm -rf *.aux *.log *.tex *.json *.out  *.png *.pdf *.pdf.png

