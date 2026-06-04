library(Matrix)
library(R.matlab)
library(UCell)
library(jsonlite)
json_data <- fromJSON("/home/e90244aa/Bleep/Major_revision/pathway_dict_msigdb_hallmark_filtered_45.json")
signatures_list <- lapply(json_data, as.character)
signatures_list <- as.list(signatures_list)
names <- c('MEND139','MEND140','MEND141','MEND142','MEND143','MEND144','MEND145','MEND146','MEND147','MEND148','MEND149','MEND150','MEND151','MEND152','MEND153',
           'MEND154','MEND156','MEND157','MEND158','MEND159','MEND160','MEND161','MEND162',
           'INT25','INT26','INT27','INT28','INT35')
for (name in names){
  link <- paste0("/home/e90244aa/Bleep/Major_revision/",name,"_data_1916.mat")
  sample <- readMat(link)
  data <- sample$x
  cell_names <- unlist(sample$cell.names)
  gene_names <- unlist(sample$gene.names)
  sparse_data <- as(data, "CsparseMatrix")
  DgCMatrix_obj <- as(sparse_data, "dgCMatrix")
  colnames(DgCMatrix_obj) <- cell_names
  gene_names <- as.vector(trimws(unlist(sample$gene.names)))
  rownames(DgCMatrix_obj) <- gene_names
  scores <- ScoreSignatures_UCell(DgCMatrix_obj, features=signatures_list,maxRank=1916)
  link <- paste0("/home/e90244aa/Bleep/Major_revision/combined_pathway_expression/",name,"_pathway expression_1916.csv")
  write.csv(scores, file = link, row.names = TRUE)
}
