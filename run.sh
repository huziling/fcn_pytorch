# arguments
Num=$1
Prefix=$2 # template_batch6

# generate tex files
python main.py --pageNum $Num --prefix $Prefix

# generate pdf files
for ((it=0; it<Num; it++))
do
    pdflatex -interaction=nonstopmode $Prefix$it.tex

done

# convert to image
for ((it=0; it<Num; it++))
do
    #echo $it
    convert -density 300 $Prefix$it.pdf -filter lagrange -distort resize 50% -background white -alpha remove -quality 100 $Prefix$it.jpg

    rm -r $Prefix$it.aux $Prefix$it.log $Prefix$it.tex $Prefix$it.pdf
done

# visualize
for ((it=0; it<Num; it++))
do
    python generate_xml.py -I "$Prefix"$it.jpg
#    python generate_xml.py -I "$Prefix"$it.jpg --visualize
done


#rm -rf ./JPEGImages ./Annotations ./SegmentationClass
#mkdir ./JPEGImages ./Annotations ./SegmentationClass
#
#rm -rf *.aux *.log *.tex *.json *.out  *.pdf
#mv *.jpg ./JPEGImages
#mv *.xml ./Annotations
#mv *.pdf.jpg ./SegmentationClass

