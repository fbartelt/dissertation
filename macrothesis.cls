\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{macrothesis}[2024/11/01 Macro Thesis Template]

% Set options for languages
\RequirePackage{ifthen}
% Option to disable glossaries
% Set default to use glossaries
\newcommand{\@useglossaries}{true}
\newcommand{\@usenatbib}{true} 
\DeclareOption{noglossaries}{%
    \renewcommand{\@useglossaries}{false}%
}
\DeclareOption{nonatbib}{%
    \renewcommand{\@usenatbib}{false}%
}

% \DeclareOption*{\PassOptionsToClass{\CurrentOption}{babel}}
% Process options
\ProcessOptions\relax

% Default language settings
\def\@mainlanguage{english} % Default main language
\def\@secondlanguage{brazil} % Default second language
% User command to set the main language
\newcommand{\setmainlanguage}[1]{%
    \def\@mainlanguage{#1}%
}

% User command to set the second language
\newcommand{\setsecondlanguage}[1]{%
    \def\@secondlanguage{#1}%
}

% \LoadClass[12pt,oneside,\@mainlanguage]{book}
\LoadClass[12pt,oneside]{book}
% Load additional packages
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

% Load Babel with secondary language only
\RequirePackage[\@mainlanguage,\@secondlanguage]{babel}
% \RequirePackage{babel}

%----------------------------------------------------------------------------------------
%	REQUIRED PACKAGES
%----------------------------------------------------------------------------------------
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{graphicx}
\RequirePackage[a4paper]{geometry}
\geometry{verbose,tmargin=3cm,bmargin=2cm,lmargin=3cm,rmargin=2cm}
\RequirePackage{setspace}
\RequirePackage{microtype}
\onehalfspacing
% \RequirePackage[pdfusetitle,
%  bookmarks=true,bookmarksnumbered=false,bookmarksopen=false,
%  breaklinks=false,pdfborder={0 0 0},pdfborderstyle={},backref=false,colorlinks=false]
%  {hyperref}
\RequirePackage[pdfusetitle,
 bookmarks=true,bookmarksnumbered=false,bookmarksopen=false,
 breaklinks=true,backref=false,colorlinks=true,allcolors=teal]
 {hyperref}

% Other packages
\RequirePackage{fancyhdr}
\RequirePackage{xcolor}
% \RequirePackage[Lenny]{fncychap}
\RequirePackage{lmodern}
\RequirePackage[lmodern]{quotchap}
% \RequirePackage{algorithmic}
\RequirePackage[nottoc]{tocbibind}
\RequirePackage{import}
\RequirePackage{caption}
% \RequirePackage{ae}
\RequirePackage{amssymb}
\RequirePackage{scrbase} % Required for handling language-dependent names of sections/document elements
% \RequirePackage{dcolumn}
% If option nonatbib is set, do not load natbib, else load
\ifthenelse{\equal{\@usenatbib}{true}}{%
    \RequirePackage[authoryear,round]{natbib}
}{%
    \PackageInfo{macrothesis}{natbib package not loaded.}%
}

% Command to store thesis title
% \newcommand{\thesistitle}[1]{\def\@thesisTitle{#1}}
\def\@thesisTitle{\@title}
% Command to store author name
% \newcommand{\authorname}[1]{\def\@thesisAuthor{#1}}
\def\@thesisAuthor{\@author}
% Command to store advisor name
\newcommand{\advisorname}[1]{\def\@thesisAdvisor{#1}}
% Command to store co-advisor names (up to 2)
\newcommand{\coadvisorname}[2][]{%
    \ifx&#1& % Check if the optional argument is empty
        \def\@thesisCoadvisorB{} % If empty, set second coadvisor to empty
    \else
        \def\@thesisCoadvisorB{#1} % Set to the provided optional argument
    \fi

    \ifx&#2& % Check if the mandatory argument is empty
        \def\@thesisCoadvisorA{} % If empty, set first coadvisor to empty
    \else
        \def\@thesisCoadvisorA{#2} % Set to the provided mandatory argument
    \fi
}
% Command to store department name
\newcommand{\departmentname}[1]{\def\@departmentName{#1}}
% Command to store the program name
\newcommand{\programname}[1]{\def\@programName{#1}}
% Command to store university name
\newcommand{\universityname}[1]{\def\@universityName{#1}}
% Command to store location (city/state)
\newcommand{\locationname}[1]{\def\@locationName{#1}}
% Command to store research group name
\newcommand{\researchgroupname}[1]{\def\@researchGroupName{#1}}
% Command to store dedication text
\newcommand{\dedication}[1]{\def\@dedicationContent{#1}}
% Command to store acknowledgments text
\newcommand{\acknowledgments}[2][\acknowledgmentsname]{%
    \def\@acknowledgmentsTitle{#1} % Set title
    \def\@acknowledgmentsContent{#2} % Set content
}

% \newcommand{\abstractname}{Abstract}
% \providecaptionname{brazil,portuguese,spanish}{\abstractname}{Resumo}

% Command for abstract in first language
\newcommand{\abstract}[2][\abstractname]{%
    \def\@abstractTitle{#1} % Set title
    \def\@abstractContent{#2} % Set content
}

% \newcommand{\acronymname}{Acronyms}
\newcommand{\notationname}{Notation}
\newcommand{\acknowledgmentsname}{Acknowledgments}
% checks if option noglossaries was set, if not, define new command
\ifthenelse{\equal{\@useglossaries}{true}}{%
    % Command to enable glossaries
    \RequirePackage[acronym,nonumberlist]{glossaries}%
    % \RequirePackage{glossaries-extra}
    % \RequirePackage{glossary-mcols}
    \newglossary[nlg]{notation}{not}{ntn}{\notationname}
    \newcommand{\newnotation}[3]{%
        \newglossaryentry{#1}{%
            type=notation,
            name={#2},
            description={#3},
            sort=def
        }
    }
    % \setlength{\glsdescwidth}{0.75\hsize}

    % Allow the symbol column to wrap text
    % \setlength{\glsentrysymbolwidth}{0.2\hsize} 
    \makenoidxglossaries%
    
}{%
    % Command to disable glossaries
    \PackageInfo{macrothesis}{glossaries package not loaded. Acronym and notation management will not be available.}%
}

\newif\ifabstractsecond % Create a new conditional flag
% Command to store abstract in second language
\newcommand{\abstractsecond}[2][\abstractname]{%
    \abstractsecondtrue % Set the flag to true
    \def\@secondAbstractTitle{#1} % Set title
    \def\@secondAbstractContent{#2} % Set content
}

% Cover Page Command
\newcommand{\maketitlepage}{%
    \thispagestyle{empty}
    \pagenumbering{roman}
    \setcounter{page}{1}

    % Research group Logo
    \includegraphics[scale=0.145]{template/logo}

    % University, Department, and Research Group
    {\small \@universityName\par}
    {\small \@programName\par}
    {\small \@researchGroupName\par}

    \vspace*{6.5cm}

    % Title
    \begin{center}
        {\Large\textbf{\expandafter\MakeUppercase\expandafter{\@thesisTitle}}}\vfill{}
    \end{center}

    % Author, Location, and Year
    \begin{center}
        \textbf{\@thesisAuthor}\\
        \@locationName\\
        \the\year
    \end{center}

    \pagebreak
}

\newcommand{\preamblepage}{
    \thispagestyle{empty}

    \begin{center}
    \textbf{\@thesisAuthor}
    \end{center}

    \vfill{}

    \begin{center}
    {\Large\textbf{\expandafter\MakeUppercase\expandafter{\@thesisTitle}}}
    \end{center}

    \vfill{}

    \begin{flushright}
    \parbox[t]{0.6\columnwidth}{%
    Dissertation submitted to the {\@programName} of {\@universityName}, in partial fulfillment of the requirements for the degree of Master in Electrical Engineering.

    \vspace{1cm}

    \textbf{Advisor:} \@thesisAdvisor %
    \ifx\@thesisCoadvisorA\empty
        % No coadvisor
    \else
        \ifx\@thesisCoadvisorB\empty
            \\ \textbf{Co-Advisor:} \@thesisCoadvisorA% Only first coadvisor
        \else
            \\ \textbf{Co-Advisor:} \@thesisCoadvisorA 
            \\ \textbf{Co-Advisor:} \@thesisCoadvisorB
        \fi
    \fi
    }
    \end{flushright}

    \vspace*{\fill}

    \begin{center}
    \@locationName\\
    \the\year
    \end{center}
    \pagebreak
    
    % Dedication page
    \begin{singlespace}
    \pagebreak{}
    \end{singlespace}

    \selectlanguage{\@secondlanguage}%
    \textbf{\thispagestyle{empty}}

    \vfill{}

    \begin{flushright}
    \begin{minipage}[t]{0.4\columnwidth}%
    \selectlanguage{\@mainlanguage}%
    \begin{singlespace}
    \begin{flushright}
    \emph{\@dedicationContent}
    \par\end{flushright}
    \end{singlespace}
    %
    \end{minipage}
    \par\end{flushright}

    \vspace{3cm}

    % Acknowledgments
    \selectlanguage{\@mainlanguage}%
    \begin{singlespace}
    \pagebreak{}
    \end{singlespace}

    \chapter*{\@acknowledgmentsTitle}
    \@acknowledgmentsContent

    % Second language Abstract
    \begin{singlespace}
    \pagebreak{}
    \end{singlespace}
    %checks wheter second abstract is empty, if not, prints it
    \ifabstractsecond
        \selectlanguage{\@secondlanguage}%
        \chapter*{\@secondAbstractTitle}
        \@secondAbstractContent
    \fi
    \begin{onehalfspace}
    \pagebreak{}
    \end{onehalfspace}

    % First language Abstract
    \selectlanguage{\@mainlanguage}%
    \chapter*{\@abstractTitle}
    \@abstractContent

    \pagebreak{}

    % Table of Contents, Figures and Tables
    \tableofcontents{}\listoffigures\listoftables

    \glossarypage
    
    \section*{\protect\pagebreak}
    \pagenumbering{arabic}
    \setcounter{page}{1}
}

\newcommand{\glossarypage}{
    % Conditionally load glossaries
    \ifthenelse{\equal{\@useglossaries}{true}}{%
        \PackageInfo{macrothesis}{Adding all acronyms and notations whether they are used or not.}% 
        \glsaddall
        \printnoidxglossary[type=acronym,sort=word]
        \printnoidxglossary[type=notation,style=long]
    }{%
        % \PackageWarning{macrothesis}{glossaries package not loaded. Acronym and notation management will not be available.}%
        % Acronyms
        \selectlanguage{\@mainlanguage}%
        \chapter*{{\normalsize\addcontentsline{toc}{chapter}{Acronyms}}\foreignlanguage{american}{Acronyms}}
        % Notation
        \chapter*{{\normalsize\addcontentsline{toc}{chapter}{Notation}}\foreignlanguage{american}{Notation}}

    }

}

%----------------------------------------------------------------------------------------
%	LOCALE SETTINGS
%----------------------------------------------------------------------------------------
% \addto\captionsenglish{\renewcommand{\notationname}{Notation}}
% \addto\captionsamerican{\renewcommand{\notationname}{Notation}}
% \addto\captionsbritish{\renewcommand{\notationname}{Notation}}
% \addto\captionsaustralian{\renewcommand{\notationname}{Notation}}
% \addto\captionscanadian{\renewcommand{\notationname}{Notation}}
% \addto\captionsnewzealand{\renewcommand{\notationname}{Notation}}
% \addto\captionsUKenglish{\renewcommand{\notationname}{Notation}}
% \addto\captionsUSenglish{\renewcommand{\notationname}{Notation}}
% \addto\captionsbrazil{\renewcommand{\notationname}{Notação}}
% \addto\captionsportuguese{\renewcommand{\notationname}{Notação}}
% \addto\captionsspanish{\renewcommand{\notationname}{Notación}}
% \addto\captionsfrench{\renewcommand{\notationname}{Notation}}
% \addto\captionsgerman{\renewcommand{\notationname}{Notation}}

% \addto\captionsenglish{\renewcommand{\acknowledgmentsname}{Acknowledgments}}
% \addto\captionsamerican{\renewcommand{\acknowledgmentsname}{Acknowledgments}}
% \addto\captionsbritish{\renewcommand{\acknowledgmentsname}{Acknowledgements}}
% \addto\captionsaustralian{\renewcommand{\acknowledgmentsname}{Acknowledgements}}
% \addto\captionscanadian{\renewcommand{\acknowledgmentsname}{Acknowledgments}}
% \addto\captionsnewzealand{\renewcommand{\acknowledgmentsname}{Acknowledgements}}
% \addto\captionsUKenglish{\renewcommand{\acknowledgmentsname}{Acknowledgements}}
% \addto\captionsUSenglish{\renewcommand{\acknowledgmentsname}{Acknowledgments}}
% \addto\captionsbrazil{\renewcommand{\acknowledgmentsname}{Agradecimentos}}
% \addto\captionsportuguese{\renewcommand{\acknowledgmentsname}{Agradecimentos}}
% \addto\captionsspanish{\renewcommand{\acknowledgmentsname}{Agradecimientos}}
% \addto\captionsfrench{\renewcommand{\acknowledgmentsname}{Remerciements}}
% \addto\captionsgerman{\renewcommand{\acknowledgmentsname}{Danksagungen}}

\providecaptionname{brazil,portuguese}{\acknowledgmentsname}{Agradecimentos}
\providecaptionname{english,american,USenglish,canadian}{\acknowledgmentsname}{Acknowledgments}
\providecaptionname{british,australian,newzealand,UKenglish}{\acknowledgmentsname}{Acknowledgements}
\providecaptionname{spanish}{\acknowledgmentsname}{Agradecimientos}
\providecaptionname{french}{\acknowledgmentsname}{Remerciements}
\providecaptionname{german,ngerman}{\acknowledgmentsname}{Danksagungen}

\providecaptionname{brazil,portuguese}{\notationname}{Notação}
\providecaptionname{english,american,USenglish,canadian,british,australian,newzealand,UKenglish,french,german,ngerman}{\notationname}{Notation}
\providecaptionname{spanish}{\notationname}{Notación}
