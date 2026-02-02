library(Matrix)
library(R.matlab)
library(UCell)
library(jsonlite)
json_data <- fromJSON("/home/e90244aa/Bleep/GBM_RAVI_etal_2/msig gbm ucell data/pathway_dic_gbm_msig_7.json")
signatures_list <- lapply(json_data, as.character)
signatures_list <- as.list(signatures_list)
names <- c("296_T","334_T","242_T","248_T","259_T")
for (name in names){
  link <- paste0("/home/e90244aa/Bleep/GBM_RAVI_etal_2/msig gbm ucell data/",name,"_data.mat")
  sample <- readMat(link)
  data <- sample$x
  cell_names <- unlist(sample$cell.names)
  gene_names <- unlist(sample$gene.names)
  sparse_data <- as(data, "CsparseMatrix")
  DgCMatrix_obj <- as(sparse_data, "dgCMatrix")
  colnames(DgCMatrix_obj) <- cell_names
  gene_names <- as.vector(trimws(unlist(sample$gene.names)))
  rownames(DgCMatrix_obj) <- gene_names
  scores <- ScoreSignatures_UCell(DgCMatrix_obj, features=signatures_list,maxRank=2106)
  link <- paste0("/home/e90244aa/Bleep/GBM_RAVI_etal_2/msig gbm ucell data/",name,"_pathway expression_2.csv")
  write.csv(scores, file = link, row.names = TRUE)
}
