// This file is part of EigenQP.
//
// EigenQP is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// EigenQP is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with EigenQP.  If not, see <http://www.gnu.org/licenses/>.

// associated header
#include "Gurobi.h"

namespace Eigen
{


/**
	*												GurobiCommon
	*/


GurobiCommon::GurobiCommon():
	Q_(),
	C_(),
	Beq_(),
	Bineq_(),
	X_(),
	fail_(0),
	nrvar_(0),
	nreq_(0),
	nrineq_(0),
	iter_(2),
	env_(),
	model_(env_)
{
}


const VectorXi& GurobiCommon::iter() const
{
	return iter_;
}


int GurobiCommon::fail() const
{
	return fail_;
}


const VectorXd& GurobiCommon::result() const
{
	return X_;
}


void GurobiCommon::problem(int nrvar, int nreq, int nrineq)
{
	for(int i = 0; i < nrvar_; ++i)
	{
	  model_.remove(*(vars_+i));
	}

	for(int i = 0; i < nreq_; ++i)
	{
	  model_.remove(*(eqconstr_+i));
	}

	for(int i = 0; i < nrineq_; ++i)
	{
	  model_.remove(*(ineqconstr_+i));
	}

	eqvars_.clear();
	ineqvars_.clear();
	lvars_.clear();
	rvars_.clear();

	nrvar_ = nrvar;
	nreq_ = nreq;
	nrineq_ = nrineq;

	Q_.resize(nrvar, nrvar);

	C_.resize(nrvar);
	Beq_.resize(nreq);
	Bineq_.resize(nrineq);
	X_.resize(nrvar);

	vars_ = model_.addVars(nrvar, GRB_CONTINUOUS);

	for(int i=0; i < Q_.size(); ++i)
	{
		lvars_.push_back(*(vars_+i/nrvar));
		rvars_.push_back(*(vars_+i%nrvar));
	}

	eqconstr_ = model_.addConstrs(nreq);
	std::vector<char> eqsense(static_cast<size_t>(nreq), '=');
	model_.set(GRB_CharAttr_Sense, eqconstr_, eqsense.data(), nreq);

	eqvars_.reserve(static_cast<size_t>(nrvar*nreq));
	for(int i = 0; i < nrvar; ++i) {
		for(int j = 0; j < nreq; ++j)
		{
			eqvars_.push_back(*(vars_+i));
		}
	}

	ineqconstr_ = model_.addConstrs(nrineq);
	std::vector<char> ineqsense(static_cast<size_t>(nrineq), '<');
	model_.set(GRB_CharAttr_Sense, ineqconstr_, ineqsense.data(), nrineq);

	for(int i = 0; i < nrvar; ++i) {
		for(int j = 0; j < nrineq; ++j)
		{
			ineqvars_.push_back(*(vars_+i));
		}
	}
}

void GurobiCommon::fillQCBf(int nreq, int nrineq,
	const MatrixXd& Q, const VectorXd& C,
	const VectorXd& Beq, const VectorXd& Bineq)
{
	Q_ = Q;
	C_ = C;

	Beq_ = Beq;
	Bineq_ = Bineq;
}


/**
	*												GurobiDense
	*/


GurobiDense::GurobiDense()
{ }


GurobiDense::GurobiDense(int nrvar, int nreq, int nrineq)
{
	problem(nrvar, nreq, nrineq);
}


void GurobiDense::problem(int nrvar, int nreq, int nrineq)
{
	GurobiCommon::problem(nrvar, nreq, nrineq);
}


bool GurobiDense::solve(const MatrixXd& Q, const VectorXd& C,
	const MatrixXd& Aeq, const VectorXd& Beq,
	const MatrixXd& Aineq, const VectorXd& Bineq,
	const VectorXd& XL, const VectorXd& XU)
{
	std::cout << "ENTER SOLVE" << std::endl;
	//Objective
	GRBQuadExpr qexpr;
	qexpr.addTerms(Q.data(), rvars_.data(), lvars_.data(), static_cast<int>(Q.size()));

	std::cout << "Linexpr" << std::endl;
	GRBLinExpr lexpr;
	std::cout << "Add terms" << std::endl;
	lexpr.addTerms(C.data(), vars_, nrvar_);
	std::cout << "Setting obj" << std::endl;
	model_.setObjective(0.5*qexpr+lexpr);

	std::cout << "Setting bounds..." << std::endl;
	model_.set(GRB_DoubleAttr_LB, vars_, XL.data(), nrvar_);
	std::cout << "XL Done" << std::endl;
	model_.set(GRB_DoubleAttr_UB, vars_, XU.data(), nrvar_);
	std::cout << "XU Done" << std::endl;

	//Update eq and ineq, column by column
	for(int i = 0; i < nrvar_; ++i)
	{
		model_.chgCoeffs(eqconstr_, eqvars_.data()+nreq_*i, Aeq.col(i).data(), static_cast<int>(Aeq.rows()));
		model_.chgCoeffs(ineqconstr_, ineqvars_.data()+nrineq_*i, Aineq.col(i).data(), static_cast<int>(Aineq.rows()));
	}

	for(int i = 0; i < nreq_; ++i)
	{
		(eqconstr_+i)->set(GRB_DoubleAttr_RHS, Beq(i));
	}

	for(int i = 0; i < nrineq_; ++i)
	{
		(ineqconstr_+i)->set(GRB_DoubleAttr_RHS, Bineq(i));
	}

	model_.optimize();

	model_.write("test.lp");

	bool success = model_.get(GRB_IntAttr_Status) == GRB_OPTIMAL;
	double* result = model_.get(GRB_DoubleAttr_X, vars_, nrvar_);
	X_ = Map<VectorXd>(result, nrvar_);

	return success;
}


/**
	*												GurobiSparse
	*/


//GurobiSparse::GurobiSparse():
//  A_(),
//  iA_()
//{ }
//
//
//GurobiSparse::GurobiSparse(int nrvar, int nreq, int nrineq):
//  A_(),
//  iA_()
//{
//  problem(nrvar, nreq, nrineq);
//}


//void GurobiSparse::problem(int nrvar, int nreq, int nrineq)
//{
	//GurobiCommon::problem(nrvar, nreq, nrineq);
//	int nrconstr = nreq + nrineq;
//
//	A_.resize(nrvar, nrconstr);
//	iA_.resize(nrvar + 1, nrconstr);
//}


//bool GurobiSparse::solve(const MatrixXd& Q, const VectorXd& C,
//	const SparseMatrix<double>& Aeq, const VectorXd& Beq,
//	const SparseMatrix<double>& Aineq, const VectorXd& Bineq,
//	bool isDecomp)
//{
//	int nrvar = static_cast<int>(C.rows());
//	int nreq = static_cast<int>(Beq.rows());
//	int nrineq = static_cast<int>(Bineq.rows());
//
//	int fddmat = static_cast<int>(Q_.rows());
//	int n = nrvar;
//	double crval;
//	int fdamat = static_cast<int>(A_.rows());
//	int q = nreq + nrineq;
//	int meq = nreq;
//	int nact;

//	fillQCBf(nreq, nrineq, Q, C, Beq, Bineq, isDecomp);
//
//	iA_.row(0).setZero();
//
//	// in A{eq,ineq} row is the constraint index, col is the variable index
//	// so row its A_ col and col is A_ row.
//	for(int k = 0; k < Aeq.outerSize(); ++k)
//	{
//		for(SparseMatrix<double>::InnerIterator it(Aeq, k); it; ++it)
//		{
//			int nrValCi = iA_(0, it.row()) += 1;
//			iA_(nrValCi, it.row()) = it.col() + 1;
//			A_(nrValCi - 1, it.row()) = it.value();
//		}
//	}
//
//	for(int k = 0; k < Aineq.outerSize(); ++k)
//	{
//		for(SparseMatrix<double>::InnerIterator it(Aineq, k); it; ++it)
//		{
//			int nrValCi = iA_(0, it.row() + nreq) += 1;
//			iA_(nrValCi, it.row() + nreq) = it.col() + 1;
//			A_(nrValCi - 1, it.row() + nreq) = -it.value();
//		}
//	}
//
//	qpgen1_(Q_.data(), C_.data(), &fddmat, &n, X_.data(), &crval,
//		A_.data(), iA_.data(), B_.data(), &fdamat, &q, &meq, iact_.data(), &nact,
//		iter_.data(), work_.data(), &fail_);
//
//	return fail_ == 0;
//}


} // namespace Eigen
