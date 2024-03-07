#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#define WORD_LENGTH 30
#define LR 0.05
#define ITER 500
#define EPOCHS 250
#define BATCH_SIZE 35
#define SENTENCE_LENGTH 700
//typedef declerations
typedef struct{
char **wordList;
int dictSize;
int wordCounter;
}Dictionary;
typedef struct{
char **sentenceList;
int listSize;
int sentenceCounter;
}SentenceList;
// function declerations
double dtanh(double x) {
double coshx = cosh(x);
return 1.0 / (coshx * coshx);
}
double predictdata(double *,double *,int );
void gradientDescent(double *,SentenceList *,Dictionary *dict);
void stochasticGradientDescent(double *W,SentenceList *sentl,Dictionary
*dict);
void ADAM(double *,SentenceList *,Dictionary *);
void printHotVector(double*,int);
void createHotVector(Dictionary*,char*,double*);
Dictionary* initializeDictionary();
SentenceList* initializeSentenceList();
bool readDictFile(Dictionary*);
bool readSentFile(SentenceList*);
bool expandDict(Dictionary*);
bool expandSent(SentenceList*);
bool isExist(char*, Dictionary*);
void printDict(Dictionary*);
void trimWord(char*);
int main(){
srand(time(NULL));
int i,totalwordCounter;
Dictionary *dict = initializeDictionary();
printf(" Reading Words for Dictionary from dataset of sentences.");
printf("\n Reading %s !",readDictFile(dict) ? "succeeded" : "failed" );
SentenceList *sentl=initializeSentenceList();
printf(" Reading each sentence from text and making a Train sentence list. ");
printf("\n Reading %s !",readSentFile(sentl) ? "succeeded" : "failed"
);
SentenceList *testsentl=initializeSentenceList();
printf(" Reading each sentence from text and making a Test sentence
list. ");
printf("\n Reading %s !",readSentFile(testsentl) ? "succeeded" :
"failed" );
double *w=(double*)calloc(dict->wordCounter,sizeof(double));
printf("\nPlease enter initial w : ");
scanf("%lf",&w[0]);
for(i=1;i<dict->wordCounter;i++)
w[i]=w[0];
clock_t start, end;
double cpu_time_used;
start = clock();
int choice= 1;
printf("\n1-Gradient Descent\n2-Stochastic Gradient Descent\n3-
ADAM\nPlease enter the method : ");
scanf("%d",&choice);
switch(choice){
case 1:
gradientDescent(w,sentl,dict);
break;
case 2:
stochasticGradientDescent(w,sentl,dict);
break;
case 3:
ADAM(w,sentl,dict);
break;
default:
gradientDescent(w,sentl,dict);
}
end = clock();
cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
printf("\nExecution time: %f seconds\n", cpu_time_used);
int success=0;
double predict;
char tempSent[SENTENCE_LENGTH];
for (i=0;i<testsentl->sentenceCounter;i++){
double *testhotvector;
testhotvector=(double*)calloc(dict->wordCounter,sizeof(double));
strcpy(tempSent,testsentl->sentenceList[i]);
createHotVector(dict,tempSent,testhotvector);
predict=predictdata(w,testhotvector,dict->wordCounter);
if(predict<-0.3&&i<20){
success+=1;
}
if(predict>0.3&&i>=20){
success+=1;
}
free(testhotvector);
}
printf("\n Your Success = %d/%d",success,testsentl->sentenceCounter);
free(dict);free(sentl);free(w);
return 0;
}
bool readDictFile(Dictionary* dict){
char _fileName[WORD_LENGTH];
char buffer[WORD_LENGTH];
FILE* f_ptr = NULL;
do{
printf("\n --> enter the file name : ");
scanf("%s",_fileName);
f_ptr = fopen(_fileName,"r");
if (f_ptr == NULL) printf("\tcouldn't opened,try again");
}while( !f_ptr );
while ( fscanf(f_ptr , "%s" , buffer) != EOF ){
trimWord(buffer);
if ( !isExist(buffer,dict) && strlen(buffer) > 0){
strcpy(dict->wordList[dict->wordCounter],buffer); // if
length of buffer is still > 0 then push it to the dictionary
dict->wordCounter++;
// checking wheter we exceeding our limits or not
if ( dict->wordCounter+1 >= dict->dictSize )
if ( !expandDict(dict) ){
fclose(f_ptr);
return false;
}
}
}
fclose(f_ptr);
return true;
}
bool readSentFile(SentenceList* sentl){
char _fileName[WORD_LENGTH];
char buffer[WORD_LENGTH];
FILE* f_ptr = NULL;
do{
printf("\n --> enter the file name : ");
scanf("%s",_fileName);
f_ptr = fopen(_fileName,"r");
if (f_ptr == NULL) printf("\tcouldn't opened,try again");
}while( !f_ptr );
char sentence[SENTENCE_LENGTH]={0};
char ch;
int i = 0;
while ( !feof(f_ptr) ){
ch = fgetc(f_ptr);
if ( ch != '.' && ch != '?' && ch != '!') { // add extra punctions
like ! ? ... etc.
if (i == 0 && ch == ' ');
else sentence[i++] = ch;
}
else{
sentence[i] = '\0';
if(strlen(sentence) > 0){
strcpy(sentl->sentenceList[sentl-
>sentenceCounter],sentence);
sentl->sentenceCounter++;
// checking wheter we exceeding our limits or not
if ( sentl->sentenceCounter+1 >= sentl->listSize )
if ( !expandSent(sentl) ){
fclose(f_ptr);
return false;
}
i = 0;
sentence[0] = '\0';
}
}
}
}
Dictionary* initializeDictionary(){
Dictionary *dict = (Dictionary*)malloc(sizeof(Dictionary));
dict->dictSize = 10;
dict->wordCounter = 0;
dict->wordList = (char**)malloc(sizeof(char*) * dict->dictSize);
int i;
for ( i = 0 ; i<dict->dictSize ; i++)
dict->wordList[i] = (char*)calloc(WORD_LENGTH,sizeof(char));
return dict;
}
SentenceList* initializeSentenceList(){
SentenceList *sentlist = (SentenceList*)malloc(sizeof(SentenceList));
sentlist->listSize=10;
sentlist->sentenceCounter=0;
sentlist->sentenceList=(char**)malloc(sizeof(char*)*sentlist->listSize);
int i;
for(i=0;i<sentlist->listSize;i++){
sentlist-
>sentenceList[i]=(char*)calloc(SENTENCE_LENGTH,sizeof(char));
}
return sentlist;
}
bool expandDict(Dictionary* dict){
dict->dictSize += 50; // adding 50 more spaces
dict->wordList = realloc(dict->wordList,sizeof(char*) * dict->dictSize);
int i;
for(i = dict->dictSize-50; i< dict->dictSize ; i++)
dict->wordList[i] = (char*)malloc(sizeof(char) * WORD_LENGTH);
return dict->wordList ? true : false;
}
bool expandSent(SentenceList* sentl){
sentl->listSize+=50; // adding 50 more spaces
sentl->sentenceList= realloc(sentl->sentenceList,sizeof(char*) * sentl-
>listSize);
int i;
for(i = sentl->listSize-50; i< sentl->listSize ; i++)
sentl->sentenceList[i] = (char*)malloc(sizeof(char) *
SENTENCE_LENGTH);
return sentl->sentenceList ? true : false;
}
bool isExist(char* word, Dictionary* dict ){
int i=0;
while ( i < dict->wordCounter )
if ( !strcmp(dict->wordList[i++],word) )
return true;
return false;
}
void printDict(Dictionary *dict){
int i;
system("cls");
printf("\n\t$ Sports Dictionary $ \t");
for( i = 0 ; i < dict->wordCounter ; i++){
printf("\n\t# Word-%02d : %s",i+1,dict->wordList[i]);
}
}
void trimWord(char* w){
int i,j,k;
// for(k = i; k<n;k++) w[k] = w[k+1];
if (strlen(w) == 1){
w[0] = '\0';
}
else
for(i = 0; i<strlen(w) ;i++){
if( isdigit(w[i]) ){
w[0] = '\0'; // NULLing the first element in case of
the word consisting of numbers (digits)
break;
}
if(ispunct(w[i])){
if( w[i] != '-'){
w[i] = '\0';
break;
}
//for(k = i; k<strlen(w);k++) w[k] = w[k+1];
else if ( !isalpha(w[i-1]) || !isalpha(w[i+1])){
w[i] = '\0';
break;
}
// for(k = i; k<strlen(w);k++) w[k] = w[k+1];
}
// if current char is a '-' and char before & after it is an
alphabetical char then keep it
// otherwise just delete it by moving chars from right to
left until that char's index.
else w[i] = tolower(w[i]); // if current char is an
alphabetical char just make it lowercase
}
}
void createHotVector(Dictionary *dict,char* sentence,double* hotVector){
int i,j,k;
char buffer[WORD_LENGTH];
for(i = 0; i<strlen(sentence) ;++i){
if(isalpha(sentence[i])){
j = i;
if(isalpha(sentence[j+1])){
while( isalpha(sentence[j]) ) {
buffer[j-i] = tolower(sentence[j]);
j++;
}
buffer[j-i] = '\0';
i = j;
// LABELING
for(k=0; k<dict->wordCounter ;k++)
if( !strcmp(dict->wordList[k],buffer)){
hotVector[k] = 1.0;
break;
}
}
}
}
}
void gradientDescent(double *W,SentenceList *sentl,Dictionary *dict){
int i,j;
int iter;
double wx=0;
double grad;
double desired;
for (iter = 0; iter < ITER; iter++) {
// kontrol et
for(j=0;j<160;j++){
if(j<80) desired=-1.00;
if(j>=80) desired=1.00;
double *sentence_hotVector;
sentence_hotVector = (double*)calloc(dict-
>wordCounter,sizeof(double));
createHotVector(dict,sentl->sentenceList[j],sentence_hotVector);
for (i = 0; i < dict->wordCounter; i++) {
wx=W[i]*sentence_hotVector[i];
grad=(double)(1.00/dict->wordCounter)*(-2*(desired-
tanh(wx))*dtanh(wx)*sentence_hotVector[i]);
W[i] =W[i] - LR*grad;
wx=0;
}
free(sentence_hotVector);
}
}
}
void stochasticGradientDescent(double *W,SentenceList *sentl,Dictionary
*dict){
int i,j,k;
int iter;
double wx=0;
double grad;
double desired;
for (iter = 0; iter < ITER; iter++) {
// kontrol et
for(k=0;k<BATCH_SIZE;k++){
j = rand() % 160;
if(j<80) desired=-1.00;
if(j>=80) desired=1.00;
double *sentence_hotVector;
sentence_hotVector = (double*)calloc(dict-
>wordCounter,sizeof(double));
createHotVector(dict,sentl->sentenceList[j],sentence_hotVector);
for (i = 0; i < dict->wordCounter; i++) {
wx=W[i]*sentence_hotVector[i];
grad=(double)(1.00/dict->wordCounter)*(-2*(desired-
tanh(wx))*dtanh(wx)*sentence_hotVector[i]);
W[i] =W[i] - LR*grad;
wx=0;
}
free(sentence_hotVector);
}
}
}
double predictdata(double *trained_W,double *testData,int dictsize){
double result=0;
double prediction;
int i;
for (i = 0; i < dictsize; i++) {
result += trained_W[i] * testData[i];
}
prediction=(double)tanh(result);
printf("\n prediction result : %f ",prediction);
if(prediction>0.3){
printf("Context of Sports Dictionary\n");
}
else if(prediction<-0.3){
printf("Context of Politics Dictionary\n");
}
else{
printf("Success rate too low to make a prediction\n");
}
return prediction;
}
void ADAM(double *W,SentenceList *sentl,Dictionary *dict){
int i,j;
double b1=0.9;
double b2=0.999;
double e=0.00000001;
double alpha=LR * sqrt(1.0 - pow(b2,1)) / (1.0 - pow(b1,1));
int epoch;
double wx=0;
double Vw=0;
double Sw=0;
double grad;
double desired;
for (epoch = 0; epoch < EPOCHS; epoch++) {
// kontrol et
for(j=0;j<160;j++){
if(j<80) desired=-1.00;
if(j>=80) desired=1.00;
double *sentence_hotVector;
sentence_hotVector = (double*)calloc(dict-
>wordCounter,sizeof(double));
createHotVector(dict,sentl->sentenceList[j],sentence_hotVector);
for (i = 0; i < dict->wordCounter; i++) {
wx=W[i]*sentence_hotVector[i];
grad=(double)(1.00/dict->wordCounter)*(-2*(desired-
tanh(wx))*dtanh(wx)*sentence_hotVector[i]);
Vw=b1*Vw+(double)(1-b1)*grad;
Sw=b2*Sw+(double)(1-b2)*(grad*grad);
W[i] =W[i] - (double)alpha*Vw/sqrt(Sw+e);
wx=0;
}
free(sentence_hotVector);
}
}
}
